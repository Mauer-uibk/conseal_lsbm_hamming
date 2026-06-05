import unittest
import numpy as np
import conseal as cl
import time

class TestHammingLSBM(unittest.TestCase):
    def test_parity_matrix(self):
        # Test known parity matrix for k=3
        H_k3 = cl.coding.hamming.generate_parity_matrix(3)
        expected_k3 = np.array([[0, 0, 0, 1, 1, 1, 1],
                                [0, 1, 1, 0, 0, 1, 1],
                                [1, 0, 1, 0, 1, 0, 1]])
        np.testing.assert_array_equal(H_k3, expected_k3)

        # Test known parity matrix for k=4
        H_k4 = cl.coding.hamming.generate_parity_matrix(4)
        expected_k4 = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                                [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 ,1],
                                [1 ,0 ,1 ,0 ,1 ,0 ,1 ,0 ,1 ,0 ,1 ,0 ,1 ,0 ,1]])
        np.testing.assert_array_equal(H_k4.astype(int), expected_k4)    

    def test_lsbm_hamming_embed(self):
        k = 3
        cover = np.array([155, 144, 133, 122, 111, 100, 99], dtype=np.int16)
        msg = np.array([1, 0, 1], dtype=np.int8)

        embed_rng = np.random.default_rng(123)
        stego = cl.coding.hamming.embed_lsbm_hamming(
            cover,
            msg,
            k,
            rng=embed_rng
        )
        expected_stego = np.array([155, 144, 133, 122, 110, 100, 99], dtype=np.int16)
        np.testing.assert_array_equal(stego, expected_stego)

    def test_lsbm_hamming_no_change(self):
        k = 3
        cover = np.array([155, 144, 133, 122, 111, 100, 99], dtype=np.int16)
        msg = np.array([0, 0, 0], dtype=np.int8)

        embed_rng = np.random.default_rng(123)
        stego = cl.coding.hamming.embed_lsbm_hamming(
            cover,
            msg,
            k,
            rng=embed_rng
        )
        np.testing.assert_array_equal(stego, cover)

        # Test with a message equal to the current syndrome -> no change
        H = cl.coding.hamming.generate_parity_matrix(k)
        syndrome = (H @ (cover % 2)) % 2
        msg2 = syndrome.astype(np.int8)
        stego2 = cl.coding.hamming.embed_lsbm_hamming(
            cover,
            msg2,
            k,
            rng=embed_rng
        )
        np.testing.assert_array_equal(stego2, cover)

    def test_hamming_simulation_vs_true(self):
        k = 3
        num_blocks = 1000
        n = 2**k - 1

        rng = np.random.default_rng(12345)
        cover = rng.integers(0, 256, size=(num_blocks, n), dtype=np.int16)
        msg = rng.integers(0, 2, size=(num_blocks, k), dtype=np.int8)

        changes_true = 0
        embed_rng = np.random.default_rng(54321)
        start_true = time.perf_counter()
        for i in range(num_blocks):
            stego = cl.coding.hamming.embed_lsbm_hamming(
                cover[i],
                msg[i],
                k,
                rng=embed_rng
            )
            changes_true += np.count_nonzero(cover[i] != stego)
        time_true = time.perf_counter() - start_true

        sim_rng = np.random.default_rng(6789)
        start_sim = time.perf_counter()
        changes_sim = cl.coding.hamming.simulate_embedding(
            k,
            num_blocks,
            rng=sim_rng
        )
        time_sim = time.perf_counter() - start_sim

        e_true = k * num_blocks / changes_true if changes_true > 0 else float('inf')
        e_sim = k * num_blocks / changes_sim if changes_sim > 0 else float('inf')
        
        # Test precision tolerance between true embedding and simulation
        self.assertAlmostEqual(e_true, e_sim, delta=0.5)

        # Simulation should be much faster than actual embedding
        self.assertLess(time_sim, time_true)

    def test_hamming_embed_extract(self):
        for k in [3, 4, 5]:
            for seed in [42, 123]:
                for msg_size in [10, 100]:
                    with self.subTest(k=k, seed=seed, msg_size=msg_size):
                        n = 2**k - 1
                        # Calculate required blocks to fit message size in bits
                        num_blocks = int(np.ceil((msg_size * 8) / k))
                        
                        rng = np.random.default_rng(seed)
                        cover = rng.integers(0, 256, size=(num_blocks, n), dtype=np.uint8)
                        
                        msg = rng.integers(0, 256, size=msg_size, dtype=np.uint8)

                        embed_rng = np.random.default_rng(seed + 1)
                        stego = cl.coding.hamming.embed_message_lsbm_hamming(
                            cover,
                            msg,
                            k,
                            rng=embed_rng
                        )

                        decoded_bits = cl.coding.hamming.extract_lsbm_hamming(
                            stego,
                            k,
                            num_bits=len(msg) * 8
                        )
                        expected_bits = np.unpackbits(msg)
                        np.testing.assert_array_equal(decoded_bits, expected_bits)

    def test_hamming_message_too_long(self):
        k = 4
        n = 2**k - 1
        # Cover is only long enough for 1 block
        cover = np.zeros(n, dtype=np.uint8)
        # Message is 2 bytes = 16 bits, requiring ceil(16/4) = 4 blocks
        msg = np.array([255, 255], dtype=np.uint8)
        
        with self.assertRaises(ValueError):
            cl.coding.hamming.embed_message_lsbm_hamming(cover, msg, k)

    def test_analytical_e(self):
        # We know for LSBM without coding replacing 1 bit/pixel, e approaches 2
        # And with Hamming codes, test calculate
        e_val = cl.coding.hamming.analytical_e(0.4)
        self.assertGreater(e_val, 2.0)
        self.assertLess(e_val, cl.coding.efficiency(0.4))
        
        # Detectability comparison: Uncoded (e=2) vs Coding
        e_uncoded = 2.0
        e_coded = cl.coding.hamming.analytical_e(3/7) # k=3
        self.assertGreater(e_coded, e_uncoded)

    def test_analytical_e_interpolation(self):
        alpha = 0.8
        e_alpha = cl.coding.hamming.analytical_e(alpha)
        _, _, _, e_k1 = cl.coding.hamming.hamming_params(1)
        _, _, _, e_k2 = cl.coding.hamming.hamming_params(2)
        self.assertGreaterEqual(e_alpha, min(e_k1, e_k2))
        self.assertLessEqual(e_alpha, max(e_k1, e_k2))
        self.assertAlmostEqual(cl.coding.hamming.analytical_e(1.0), e_k1, delta=1e-6)

if __name__ == '__main__':
    unittest.main()

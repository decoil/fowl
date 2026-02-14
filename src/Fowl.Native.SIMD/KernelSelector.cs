using System;
using System.Runtime.Intrinsics.X86;

namespace Fowl.Native.SIMD
{
    /// <summary>
    /// Automatically selects the best available SIMD implementation at runtime.
    /// Checks for AVX2, SSE2 support and falls back to scalar if neither is available.
    /// </summary>
    public static class KernelSelector
    {
        #region Hardware Detection

        /// <summary>
        /// Gets whether AVX2 is supported on this hardware.
        /// </summary>
        public static bool IsAvx2Supported { get; } = Avx2.IsSupported;

        /// <summary>
        /// Gets whether SSE2 is supported on this hardware.
        /// </summary>
        public static bool IsSse2Supported { get; } = Sse2.IsSupported;

        /// <summary>
        /// Gets the name of the best available SIMD implementation.
        /// </summary>
        public static string BestImplementation { get; } = GetBestImplementation();

        /// <summary>
        /// Minimum array size to benefit from SIMD overhead.
        /// Arrays smaller than this use scalar operations.
        /// </summary>
        public const int SimdThreshold = 16;

        private static string GetBestImplementation()
        {
            if (IsAvx2Supported) return "AVX2";
            if (IsSse2Supported) return "SSE2";
            return "Scalar";
        }

        /// <summary>
        /// Prints SIMD capability information to console.
        /// </summary>
        public static void PrintSimdInfo()
        {
            Console.WriteLine("=== Fowl.Native.SIMD Hardware Detection ===");
            Console.WriteLine($"AVX2 Supported: {IsAvx2Supported}");
            Console.WriteLine($"SSE2 Supported: {IsSse2Supported}");
            Console.WriteLine($"Best Implementation: {BestImplementation}");
            Console.WriteLine($"SIMD Threshold: {SimdThreshold} elements");
            Console.WriteLine("==========================================");
        }

        #endregion

        #region Element-wise Operations

        /// <summary>
        /// Element-wise addition with automatic implementation selection.
        /// </summary>
        public static void Add(double[] a, double[] b, double[] result)
        {
            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                ScalarAdd(a, b, result);
            }
            else if (IsAvx2Supported)
            {
                Avx2Kernels.Add(a, b, result);
            }
            else
            {
                Sse2Kernels.Add(a, b, result);
            }
        }

        /// <summary>
        /// Element-wise subtraction with automatic implementation selection.
        /// </summary>
        public static void Subtract(double[] a, double[] b, double[] result)
        {
            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                ScalarSubtract(a, b, result);
            }
            else if (IsAvx2Supported)
            {
                Avx2Kernels.Subtract(a, b, result);
            }
            else
            {
                Sse2Kernels.Subtract(a, b, result);
            }
        }

        /// <summary>
        /// Element-wise multiplication with automatic implementation selection.
        /// </summary>
        public static void Multiply(double[] a, double[] b, double[] result)
        {
            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                ScalarMultiply(a, b, result);
            }
            else if (IsAvx2Supported)
            {
                Avx2Kernels.Multiply(a, b, result);
            }
            else
            {
                Sse2Kernels.Multiply(a, b, result);
            }
        }

        /// <summary>
        /// Element-wise division with automatic implementation selection.
        /// </summary>
        public static void Divide(double[] a, double[] b, double[] result)
        {
            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                ScalarDivide(a, b, result);
            }
            else if (IsAvx2Supported)
            {
                Avx2Kernels.Divide(a, b, result);
            }
            else
            {
                Sse2Kernels.Divide(a, b, result);
            }
        }

        /// <summary>
        /// Add scalar with automatic implementation selection.
        /// </summary>
        public static void AddScalar(double[] a, double scalar, double[] result)
        {
            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                for (int i = 0; i < a.Length; i++)
                    result[i] = a[i] + scalar;
            }
            else if (IsAvx2Supported)
            {
                Avx2Kernels.AddScalar(a, scalar, result);
            }
            else
            {
                Sse2Kernels.AddScalar(a, scalar, result);
            }
        }

        /// <summary>
        /// Multiply by scalar with automatic implementation selection.
        /// </summary>
        public static void MultiplyScalar(double[] a, double scalar, double[] result)
        {
            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                for (int i = 0; i < a.Length; i++)
                    result[i] = a[i] * scalar;
            }
            else if (IsAvx2Supported)
            {
                Avx2Kernels.MultiplyScalar(a, scalar, result);
            }
            else
            {
                Sse2Kernels.MultiplyScalar(a, scalar, result);
            }
        }

        /// <summary>
        /// Negate with automatic implementation selection.
        /// </summary>
        public static void Negate(double[] a, double[] result)
        {
            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                for (int i = 0; i < a.Length; i++)
                    result[i] = -a[i];
            }
            else if (IsAvx2Supported)
            {
                Avx2Kernels.Negate(a, result);
            }
            else
            {
                Sse2Kernels.Negate(a, result);
            }
        }

        #endregion

        #region Single-Precision Operations

        /// <summary>
        /// Element-wise addition (float) with automatic implementation selection.
        /// </summary>
        public static void Add(float[] a, float[] b, float[] result)
        {
            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                for (int i = 0; i < a.Length; i++)
                    result[i] = a[i] + b[i];
            }
            else if (IsAvx2Supported)
            {
                Avx2Kernels.Add(a, b, result);
            }
            else
            {
                Sse2Kernels.Add(a, b, result);
            }
        }

        /// <summary>
        /// Element-wise multiplication (float) with automatic implementation selection.
        /// </summary>
        public static void Multiply(float[] a, float[] b, float[] result)
        {
            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                for (int i = 0; i < a.Length; i++)
                    result[i] = a[i] * b[i];
            }
            else if (IsAvx2Supported)
            {
                Avx2Kernels.Multiply(a, b, result);
            }
            else
            {
                Sse2Kernels.Multiply(a, b, result);
            }
        }

        #endregion

        #region Reduction Operations

        /// <summary>
        /// Sum all elements with automatic implementation selection.
        /// </summary>
        public static double Sum(double[] a)
        {
            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                return ScalarSum(a);
            }
            else if (IsAvx2Supported)
            {
                return Avx2Kernels.Sum(a);
            }
            else
            {
                return Sse2Kernels.Sum(a);
            }
        }

        /// <summary>
        /// Dot product with automatic implementation selection.
        /// </summary>
        public static double Dot(double[] a, double[] b)
        {
            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                return ScalarDot(a, b);
            }
            else if (IsAvx2Supported)
            {
                return Avx2Kernels.Dot(a, b);
            }
            else
            {
                return Sse2Kernels.Dot(a, b);
            }
        }

        /// <summary>
        /// Find minimum with automatic implementation selection.
        /// </summary>
        public static double Min(double[] a)
        {
            if (a.Length == 0)
                throw new ArgumentException("Array cannot be empty", nameof(a));

            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                return ScalarMin(a);
            }
            else if (IsAvx2Supported)
            {
                return Avx2Kernels.Min(a);
            }
            else
            {
                return Sse2Kernels.Min(a);
            }
        }

        /// <summary>
        /// Find maximum with automatic implementation selection.
        /// </summary>
        public static double Max(double[] a)
        {
            if (a.Length == 0)
                throw new ArgumentException("Array cannot be empty", nameof(a));

            if (a.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                return ScalarMax(a);
            }
            else if (IsAvx2Supported)
            {
                return Avx2Kernels.Max(a);
            }
            else
            {
                return Sse2Kernels.Max(a);
            }
        }

        #endregion

        #region In-Place Operations

        /// <summary>
        /// Add in-place with automatic implementation selection.
        /// </summary>
        public static void AddInPlace(double[] result, double[] b)
        {
            if (result.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                for (int i = 0; i < result.Length; i++)
                    result[i] += b[i];
            }
            else if (IsAvx2Supported)
            {
                Avx2Kernels.AddInPlace(result, b);
            }
            else
            {
                Sse2Kernels.AddInPlace(result, b);
            }
        }

        /// <summary>
        /// Multiply in-place by scalar with automatic implementation selection.
        /// </summary>
        public static void MultiplyInPlace(double[] result, double scalar)
        {
            if (result.Length < SimdThreshold || (!IsAvx2Supported && !IsSse2Supported))
            {
                for (int i = 0; i < result.Length; i++)
                    result[i] *= scalar;
            }
            else if (IsAvx2Supported)
            {
                Avx2Kernels.MultiplyInPlace(result, scalar);
            }
            else
            {
                Sse2Kernels.MultiplyInPlace(result, scalar);
            }
        }

        #endregion

        #region Scalar Fallbacks

        private static void ScalarAdd(double[] a, double[] b, double[] result)
        {
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] + b[i];
        }

        private static void ScalarSubtract(double[] a, double[] b, double[] result)
        {
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] - b[i];
        }

        private static void ScalarMultiply(double[] a, double[] b, double[] result)
        {
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] * b[i];
        }

        private static void ScalarDivide(double[] a, double[] b, double[] result)
        {
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] / b[i];
        }

        private static double ScalarSum(double[] a)
        {
            double sum = 0.0;
            for (int i = 0; i < a.Length; i++)
                sum += a[i];
            return sum;
        }

        private static double ScalarDot(double[] a, double[] b)
        {
            double sum = 0.0;
            for (int i = 0; i < a.Length; i++)
                sum += a[i] * b[i];
            return sum;
        }

        private static double ScalarMin(double[] a)
        {
            double min = a[0];
            for (int i = 1; i < a.Length; i++)
                if (a[i] < min) min = a[i];
            return min;
        }

        private static double ScalarMax(double[] a)
        {
            double max = a[0];
            for (int i = 1; i < a.Length; i++)
                if (a[i] > max) max = a[i];
            return max;
        }

        #endregion
    }
}

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Fowl.Native.SIMD
{
    /// <summary>
    /// AVX2-optimized SIMD kernels for x86/x64 processors.
    /// Uses 256-bit vectors (4 doubles or 8 floats).
    /// </summary>
    public static class Avx2Kernels
    {
        #region Double-Precision Operations

        /// <summary>
        /// Element-wise addition of two double arrays using AVX2.
        /// </summary>
        /// <param name="a">First array</param>
        /// <param name="b">Second array</param>
        /// <param name="result">Result array (must be pre-allocated)</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(double[] a, double[] b, double[] result)
        {
            int vecSize = 4; // 256 bits / 64 bits per double
            int i = 0;

            // Main AVX2 loop - process 4 elements at a time
            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                var vb = Vector256.LoadUnsafe(ref b[i]);
                var vr = Avx2.Add(va, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            // Scalar remainder
            for (; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        /// <summary>
        /// Element-wise subtraction of two double arrays using AVX2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Subtract(double[] a, double[] b, double[] result)
        {
            int vecSize = 4;
            int i = 0;

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                var vb = Vector256.LoadUnsafe(ref b[i]);
                var vr = Avx2.Subtract(va, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] - b[i];
            }
        }

        /// <summary>
        /// Element-wise multiplication of two double arrays using AVX2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply(double[] a, double[] b, double[] result)
        {
            int vecSize = 4;
            int i = 0;

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                var vb = Vector256.LoadUnsafe(ref b[i]);
                var vr = Avx2.Multiply(va, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        /// <summary>
        /// Element-wise division of two double arrays using AVX2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Divide(double[] a, double[] b, double[] result)
        {
            int vecSize = 4;
            int i = 0;

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                var vb = Vector256.LoadUnsafe(ref b[i]);
                var vr = Avx2.Divide(va, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] / b[i];
            }
        }

        /// <summary>
        /// Add scalar to each element using AVX2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddScalar(double[] a, double scalar, double[] result)
        {
            int vecSize = 4;
            int i = 0;
            var vscalar = Vector256.Create(scalar);

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                var vr = Avx2.Add(va, vscalar);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] + scalar;
            }
        }

        /// <summary>
        /// Multiply each element by scalar using AVX2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MultiplyScalar(double[] a, double scalar, double[] result)
        {
            int vecSize = 4;
            int i = 0;
            var vscalar = Vector256.Create(scalar);

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                var vr = Avx2.Multiply(va, vscalar);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] * scalar;
            }
        }

        /// <summary>
        /// Negate all elements using AVX2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Negate(double[] a, double[] result)
        {
            int vecSize = 4;
            int i = 0;
            var zero = Vector256.Create(0.0);

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                var vr = Avx2.Subtract(zero, va);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = -a[i];
            }
        }

        #endregion

        #region Single-Precision Operations

        /// <summary>
        /// Element-wise addition of two float arrays using AVX2.
        /// Processes 8 elements at a time.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(float[] a, float[] b, float[] result)
        {
            int vecSize = 8; // 256 bits / 32 bits per float
            int i = 0;

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                var vb = Vector256.LoadUnsafe(ref b[i]);
                var vr = Avx2.Add(va, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        /// <summary>
        /// Element-wise multiplication of two float arrays using AVX2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply(float[] a, float[] b, float[] result)
        {
            int vecSize = 8;
            int i = 0;

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                var vb = Vector256.LoadUnsafe(ref b[i]);
                var vr = Avx2.Multiply(va, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        #endregion

        #region Reduction Operations

        /// <summary>
        /// Sum all elements using AVX2.
        /// Uses vectorized accumulation with horizontal reduction.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Sum(double[] a)
        {
            int vecSize = 4;
            int i = 0;
            var vecSum = Vector256.Create(0.0);

            // Vectorized accumulation
            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                vecSum = Avx2.Add(vecSum, va);
            }

            // Horizontal reduction
            double total = 0.0;
            for (int j = 0; j < vecSize; j++)
            {
                total += vecSum.GetElement(j);
            }

            // Remainder
            for (; i < a.Length; i++)
            {
                total += a[i];
            }

            return total;
        }

        /// <summary>
        /// Dot product using AVX2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Dot(double[] a, double[] b)
        {
            int vecSize = 4;
            int i = 0;
            var vecSum = Vector256.Create(0.0);

            // Vectorized: multiply then accumulate
            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                var vb = Vector256.LoadUnsafe(ref b[i]);
                vecSum = Avx2.Add(vecSum, Avx2.Multiply(va, vb));
            }

            // Horizontal reduction
            double total = 0.0;
            for (int j = 0; j < vecSize; j++)
            {
                total += vecSum.GetElement(j);
            }

            // Remainder
            for (; i < a.Length; i++)
            {
                total += a[i] * b[i];
            }

            return total;
        }

        /// <summary>
        /// Find minimum element using AVX2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Min(double[] a)
        {
            if (a.Length == 0)
                throw new ArgumentException("Array cannot be empty", nameof(a));

            int vecSize = 4;
            int i = vecSize;
            var vecMin = Vector256.LoadUnsafe(ref a[0]);

            // Vectorized min
            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                vecMin = Avx2.Min(vecMin, va);
            }

            // Horizontal reduction
            double minimum = vecMin.GetElement(0);
            for (int j = 1; j < vecSize; j++)
            {
                minimum = Math.Min(minimum, vecMin.GetElement(j));
            }

            // Remainder
            for (; i < a.Length; i++)
            {
                minimum = Math.Min(minimum, a[i]);
            }

            return minimum;
        }

        /// <summary>
        /// Find maximum element using AVX2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Max(double[] a)
        {
            if (a.Length == 0)
                throw new ArgumentException("Array cannot be empty", nameof(a));

            int vecSize = 4;
            int i = vecSize;
            var vecMax = Vector256.LoadUnsafe(ref a[0]);

            // Vectorized max
            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector256.LoadUnsafe(ref a[i]);
                vecMax = Avx2.Max(vecMax, va);
            }

            // Horizontal reduction
            double maximum = vecMax.GetElement(0);
            for (int j = 1; j < vecSize; j++)
            {
                maximum = Math.Max(maximum, vecMax.GetElement(j));
            }

            // Remainder
            for (; i < a.Length; i++)
            {
                maximum = Math.Max(maximum, a[i]);
            }

            return maximum;
        }

        #endregion

        #region In-Place Operations

        /// <summary>
        /// Add arrays in-place (result = result + b).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddInPlace(double[] result, double[] b)
        {
            int vecSize = 4;
            int i = 0;

            for (; i <= result.Length - vecSize; i += vecSize)
            {
                var vr = Vector256.LoadUnsafe(ref result[i]);
                var vb = Vector256.LoadUnsafe(ref b[i]);
                vr = Avx2.Add(vr, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < result.Length; i++)
            {
                result[i] += b[i];
            }
        }

        /// <summary>
        /// Multiply in-place by scalar.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MultiplyInPlace(double[] result, double scalar)
        {
            int vecSize = 4;
            int i = 0;
            var vscalar = Vector256.Create(scalar);

            for (; i <= result.Length - vecSize; i += vecSize)
            {
                var vr = Vector256.LoadUnsafe(ref result[i]);
                vr = Avx2.Multiply(vr, vscalar);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < result.Length; i++)
            {
                result[i] *= scalar;
            }
        }

        #endregion
    }
}

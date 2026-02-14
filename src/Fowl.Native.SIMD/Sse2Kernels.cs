using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Fowl.Native.SIMD
{
    /// <summary>
    /// SSE2-optimized SIMD kernels for older x86/x64 processors.
    /// Uses 128-bit vectors (2 doubles or 4 floats).
    /// Fallback when AVX2 is not available.
    /// </summary>
    public static class Sse2Kernels
    {
        #region Double-Precision Operations

        /// <summary>
        /// Element-wise addition using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(double[] a, double[] b, double[] result)
        {
            int vecSize = 2; // 128 bits / 64 bits per double
            int i = 0;

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                var vb = Vector128.LoadUnsafe(ref b[i]);
                var vr = Sse2.Add(va, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        /// <summary>
        /// Element-wise subtraction using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Subtract(double[] a, double[] b, double[] result)
        {
            int vecSize = 2;
            int i = 0;

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                var vb = Vector128.LoadUnsafe(ref b[i]);
                var vr = Sse2.Subtract(va, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] - b[i];
            }
        }

        /// <summary>
        /// Element-wise multiplication using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply(double[] a, double[] b, double[] result)
        {
            int vecSize = 2;
            int i = 0;

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                var vb = Vector128.LoadUnsafe(ref b[i]);
                var vr = Sse2.Multiply(va, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        /// <summary>
        /// Element-wise division using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Divide(double[] a, double[] b, double[] result)
        {
            int vecSize = 2;
            int i = 0;

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                var vb = Vector128.LoadUnsafe(ref b[i]);
                var vr = Sse2.Divide(va, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] / b[i];
            }
        }

        /// <summary>
        /// Add scalar using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddScalar(double[] a, double scalar, double[] result)
        {
            int vecSize = 2;
            int i = 0;
            var vscalar = Vector128.Create(scalar);

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                var vr = Sse2.Add(va, vscalar);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] + scalar;
            }
        }

        /// <summary>
        /// Multiply by scalar using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MultiplyScalar(double[] a, double scalar, double[] result)
        {
            int vecSize = 2;
            int i = 0;
            var vscalar = Vector128.Create(scalar);

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                var vr = Sse2.Multiply(va, vscalar);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] * scalar;
            }
        }

        /// <summary>
        /// Negate using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Negate(double[] a, double[] result)
        {
            int vecSize = 2;
            int i = 0;
            var zero = Vector128.Create(0.0);

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                var vr = Sse2.Subtract(zero, va);
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
        /// Element-wise addition using SSE2 (4 floats at a time).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(float[] a, float[] b, float[] result)
        {
            int vecSize = 4;
            int i = 0;

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                var vb = Vector128.LoadUnsafe(ref b[i]);
                var vr = Sse2.Add(va, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        /// <summary>
        /// Element-wise multiplication using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply(float[] a, float[] b, float[] result)
        {
            int vecSize = 4;
            int i = 0;

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                var vb = Vector128.LoadUnsafe(ref b[i]);
                var vr = Sse2.Multiply(va, vb);
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
        /// Sum all elements using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Sum(double[] a)
        {
            int vecSize = 2;
            int i = 0;
            var vecSum = Vector128.Create(0.0);

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                vecSum = Sse2.Add(vecSum, va);
            }

            double total = vecSum.GetElement(0) + vecSum.GetElement(1);

            for (; i < a.Length; i++)
            {
                total += a[i];
            }

            return total;
        }

        /// <summary>
        /// Dot product using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Dot(double[] a, double[] b)
        {
            int vecSize = 2;
            int i = 0;
            var vecSum = Vector128.Create(0.0);

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                var vb = Vector128.LoadUnsafe(ref b[i]);
                vecSum = Sse2.Add(vecSum, Sse2.Multiply(va, vb));
            }

            double total = vecSum.GetElement(0) + vecSum.GetElement(1);

            for (; i < a.Length; i++)
            {
                total += a[i] * b[i];
            }

            return total;
        }

        /// <summary>
        /// Find minimum using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Min(double[] a)
        {
            if (a.Length == 0)
                throw new ArgumentException("Array cannot be empty", nameof(a));

            int vecSize = 2;
            int i = vecSize;
            var vecMin = Vector128.LoadUnsafe(ref a[0]);

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                vecMin = Sse2.Min(vecMin, va);
            }

            double minimum = Math.Min(vecMin.GetElement(0), vecMin.GetElement(1));

            for (; i < a.Length; i++)
            {
                minimum = Math.Min(minimum, a[i]);
            }

            return minimum;
        }

        /// <summary>
        /// Find maximum using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Max(double[] a)
        {
            if (a.Length == 0)
                throw new ArgumentException("Array cannot be empty", nameof(a));

            int vecSize = 2;
            int i = vecSize;
            var vecMax = Vector128.LoadUnsafe(ref a[0]);

            for (; i <= a.Length - vecSize; i += vecSize)
            {
                var va = Vector128.LoadUnsafe(ref a[i]);
                vecMax = Sse2.Max(vecMax, va);
            }

            double maximum = Math.Max(vecMax.GetElement(0), vecMax.GetElement(1));

            for (; i < a.Length; i++)
            {
                maximum = Math.Max(maximum, a[i]);
            }

            return maximum;
        }

        #endregion

        #region In-Place Operations

        /// <summary>
        /// Add in-place using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddInPlace(double[] result, double[] b)
        {
            int vecSize = 2;
            int i = 0;

            for (; i <= result.Length - vecSize; i += vecSize)
            {
                var vr = Vector128.LoadUnsafe(ref result[i]);
                var vb = Vector128.LoadUnsafe(ref b[i]);
                vr = Sse2.Add(vr, vb);
                vr.StoreUnsafe(ref result[i]);
            }

            for (; i < result.Length; i++)
            {
                result[i] += b[i];
            }
        }

        /// <summary>
        /// Multiply in-place by scalar using SSE2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MultiplyInPlace(double[] result, double scalar)
        {
            int vecSize = 2;
            int i = 0;
            var vscalar = Vector128.Create(scalar);

            for (; i <= result.Length - vecSize; i += vecSize)
            {
                var vr = Vector128.LoadUnsafe(ref result[i]);
                vr = Sse2.Multiply(vr, vscalar);
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

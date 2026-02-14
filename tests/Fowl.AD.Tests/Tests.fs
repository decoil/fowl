module Fowl.Tests.AD

open Expecto
open Fowl.AD

/// Helper for approximate float equality
let approxEqual (tol: float) (actual: float) (expected: float) =
    abs (actual - expected) < tol

let tests =
    testList "Algorithmic Differentiation" [
        // ========================================================================
        // Forward Mode Tests
        // ========================================================================
        
        test "diff computes derivative of sin" {
            let f x = sin x
            let value, deriv = diff' f (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt value) 0.0 "sin(0) = 0"
            Expect.floatClose Accuracy.medium (unpack_flt deriv) 1.0 "sin'(0) = cos(0) = 1"
        }
        
        test "diff computes derivative of cos" {
            let f x = cos x
            let value, deriv = diff' f (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt value) 1.0 "cos(0) = 1"
            Expect.floatClose Accuracy.medium (unpack_flt deriv) 0.0 "cos'(0) = -sin(0) = 0"
        }
        
        test "diff computes derivative of exp" {
            let f x = exp x
            let value, deriv = diff' f (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt value) 1.0 "exp(0) = 1"
            Expect.floatClose Accuracy.medium (unpack_flt deriv) 1.0 "exp'(0) = exp(0) = 1"
        }
        
        test "diff computes derivative of log" {
            let f x = log x
            let value, deriv = diff' f (pack_flt 1.0)
            Expect.floatClose Accuracy.medium (unpack_flt value) 0.0 "log(1) = 0"
            Expect.floatClose Accuracy.medium (unpack_flt deriv) 1.0 "log'(1) = 1/1 = 1"
        }
        
        test "diff computes derivative of polynomial" {
            // f(x) = x^3, f'(x) = 3x^2, f'(2) = 12
            let f x = pow x (pack_flt 3.0)
            let _, deriv = diff' f (pack_flt 2.0)
            Expect.floatClose Accuracy.medium (unpack_flt deriv) 12.0 "(x^3)' at 2 = 12"
        }
        
        test "diffF convenience function" {
            let f x = sin x
            let deriv = diffF f 0.0
            Expect.floatClose Accuracy.medium deriv 1.0 "sin'(0) = 1"
        }
        
        // ========================================================================
        // Chain Rule Tests
        // ========================================================================
        
        test "chain rule: sin(x^2)" {
            // f(x) = sin(x^2), f'(x) = 2x * cos(x^2)
            // f'(0) = 0
            let f x = sin (pow x (pack_flt 2.0))
            let deriv = diff f (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt deriv) 0.0 "chain rule at 0"
        }
        
        test "chain rule: exp(sin(x))" {
            // f(x) = exp(sin(x)), f'(x) = exp(sin(x)) * cos(x)
            // f'(0) = exp(0) * 1 = 1
            let f x = exp (sin x)
            let deriv = diff f (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt deriv) 1.0 "exp(sin(x))' at 0"
        }
        
        // ========================================================================
        // Product and Quotient Rules
        // ========================================================================
        
        test "product rule: x * sin(x)" {
            // f(x) = x * sin(x)
            // f'(x) = sin(x) + x * cos(x)
            // f'(0) = 0 + 0 = 0
            let f x = mul x (sin x)
            let deriv = diff f (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt deriv) 0.0 "product rule at 0"
        }
        
        test "product rule: x * exp(x)" {
            // f(x) = x * exp(x)
            // f'(x) = exp(x) + x * exp(x) = exp(x) * (1 + x)
            // f'(1) = e * 2 ≈ 5.436
            let f x = mul x (exp x)
            let deriv = diff f (pack_flt 1.0)
            let expected = Math.Exp(1.0) * 2.0
            Expect.floatClose Accuracy.medium (unpack_flt deriv) expected "product rule at 1"
        }
        
        test "quotient rule: sin(x)/x" {
            // f(x) = sin(x)/x
            // f'(x) = (x*cos(x) - sin(x))/x^2
            // Using L'Hopital's rule: f'(0) = 0
            let f x = div (sin x) x
            let deriv = diff f (pack_flt 0.1)  // Small but not zero
            // Approximate: derivative should be close to 0 at small x
            Expect.isTrue (abs (unpack_flt deriv) < 0.5) "quotient rule near 0"
        }
        
        // ========================================================================
        // Higher-Order Derivatives
        // ========================================================================
        
        test "second derivative of x^2" {
            // f(x) = x^2, f'(x) = 2x, f''(x) = 2
            let f x = pow x (pack_flt 2.0)
            let f' = diff f
            let f'' = diff f'
            let d2 = f'' (pack_flt 1.0)
            Expect.floatClose Accuracy.medium (unpack_flt d2) 2.0 "second derivative of x^2"
        }
        
        test "second derivative of sin" {
            // f(x) = sin(x), f'(x) = cos(x), f''(x) = -sin(x)
            // f''(0) = 0
            let f x = sin x
            let f' = diff f
            let f'' = diff f'
            let d2 = f'' (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt d2) 0.0 "second derivative of sin at 0"
        }
        
        test "third derivative of x^3" {
            // f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x, f'''(x) = 6
            let f x = pow x (pack_flt 3.0)
            let f' = diff f
            let f'' = diff f'
            let f''' = diff f''
            let d3 = f''' (pack_flt 2.0)
            Expect.floatClose Accuracy.medium (unpack_flt d3) 6.0 "third derivative of x^3"
        }
        
        // ========================================================================
        // Reverse Mode (Gradient) Tests
        // ========================================================================
        
        test "grad computes gradient of x^2" {
            // f(x) = x^2, grad f = 2x, grad at 3 = 6
            let f x = pow x (pack_flt 2.0)
            let g = grad f (pack_flt 3.0)
            Expect.floatClose Accuracy.medium (unpack_flt g) 6.0 "gradient of x^2 at 3"
        }
        
        test "gradF convenience function" {
            let f x = pow x (pack_flt 2.0)
            let g = gradF f 3.0
            Expect.floatClose Accuracy.medium g 6.0 "gradF of x^2 at 3"
        }
        
        test "grad computes gradient of sin" {
            // f(x) = sin(x), grad f = cos(x), grad at 0 = 1
            let f x = sin x
            let g = grad f (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt g) 1.0 "gradient of sin at 0"
        }
        
        // ========================================================================
        // Hessian Tests
        // ========================================================================
        
        test "hessian of x^2" {
            // f(x) = x^2, H = f'' = 2
            let f x = pow x (pack_flt 2.0)
            let h = hessian f (pack_flt 1.0)
            Expect.floatClose Accuracy.medium (unpack_flt h) 2.0 "hessian of x^2"
        }
        
        test "hessian of sin" {
            // f(x) = sin(x), H = -sin(x), at 0 = 0
            let f x = sin x
            let h = hessian f (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt h) 0.0 "hessian of sin at 0"
        }
        
        // ========================================================================
        // Laplacian Tests
        // ========================================================================
        
        test "laplacian of x^2" {
            // For 1D, Laplacian = f''
            // f(x) = x^2, laplacian = 2
            let f x = pow x (pack_flt 2.0)
            let lap = laplacian f (pack_flt 1.0)
            Expect.floatClose Accuracy.medium (unpack_flt lap) 2.0 "laplacian of x^2"
        }
        
        // ========================================================================
        // Combined Operations Tests
        // ========================================================================
        
        test "complex function: x^2 + sin(x) + exp(x)" {
            // f(x) = x^2 + sin(x) + exp(x)
            // f'(x) = 2x + cos(x) + exp(x)
            // f'(0) = 0 + 1 + 1 = 2
            let f x = add (pow x (pack_flt 2.0)) (add (sin x) (exp x))
            let deriv = diff f (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt deriv) 2.0 "complex function derivative"
        }
        
        test "nested operations: sin(exp(x))" {
            // f(x) = sin(exp(x))
            // f'(x) = cos(exp(x)) * exp(x)
            // f'(0) = cos(1) * 1 ≈ 0.540
            let f x = sin (exp x)
            let deriv = diff f (pack_flt 0.0)
            let expected = Math.Cos(1.0)
            Expect.floatClose Accuracy.medium (unpack_flt deriv) expected "nested operations"
        }
        
        // ========================================================================
        // Type Tests
        // ========================================================================
        
        test "isConstant correctly identifies constants" {
            let c = pack_flt 5.0
            Expect.isTrue (isConstant c) "pack_flt should be constant"
            
            let x = make_forward (pack_flt 5.0) (pack_flt 1.0) 1
            Expect.isFalse (isConstant x) "make_forward should not be constant"
        }
        
        test "isForward correctly identifies forward mode" {
            let x = make_forward (pack_flt 5.0) (pack_flt 1.0) 1
            Expect.isTrue (isForward x) "make_forward should be forward mode"
            
            let c = pack_flt 5.0
            Expect.isFalse (isForward c) "constant should not be forward mode"
        }
        
        // ========================================================================
        // Edge Cases
        // ========================================================================
        
        test "derivative of constant is zero" {
            let f x = pack_flt 5.0
            let deriv = diff f (pack_flt 10.0)
            Expect.floatClose Accuracy.medium (unpack_flt deriv) 0.0 "derivative of constant"
        }
        
        test "derivative of identity is one" {
            let f x = x
            let deriv = diff f (pack_flt 5.0)
            Expect.floatClose Accuracy.medium (unpack_flt deriv) 1.0 "derivative of identity"
        }
    ]

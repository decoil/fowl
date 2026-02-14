module Fowl.Tests.AD

open Expecto
open Fowl.AD

let tests =
    testList "Algorithmic Differentiation" [
        test "diff computes derivative of sin" {
            let f x = sin x
            let d = diff f (pack_flt 0.0)
            // d/dx sin(x) = cos(x), at x=0: cos(0) = 1
            Expect.floatClose Accuracy.medium (unpack_flt d) 1.0 "Derivative of sin at 0 should be 1"
        }
        
        test "diff computes derivative of cos" {
            let f x = cos x
            let d = diff f (pack_flt 0.0)
            // d/dx cos(x) = -sin(x), at x=0: -sin(0) = 0
            Expect.floatClose Accuracy.medium (unpack_flt d) 0.0 "Derivative of cos at 0 should be 0"
        }
        
        test "diff computes derivative of exp" {
            let f x = exp x
            let d = diff f (pack_flt 0.0)
            // d/dx exp(x) = exp(x), at x=0: exp(0) = 1
            Expect.floatClose Accuracy.medium (unpack_flt d) 1.0 "Derivative of exp at 0 should be 1"
        }
        
        test "diff computes derivative of polynomial" {
            // f(x) = x^3, f'(x) = 3x^2, f'(2) = 12
            let f x = pow x (pack_flt 3.0)
            let d = diff f (pack_flt 2.0)
            Expect.floatClose Accuracy.medium (unpack_flt d) 12.0 "Derivative of x^3 at 2 should be 12"
        }
        
        test "diff computes second derivative" {
            // f(x) = x^2, f'(x) = 2x, f''(x) = 2
            let f x = pow x (pack_flt 2.0)
            let f' = diff f
            let f'' = diff f'
            let d2 = f'' (pack_flt 1.0)
            Expect.floatClose Accuracy.medium (unpack_flt d2) 2.0 "Second derivative of x^2 should be 2"
        }
        
        test "grad computes gradient" {
            // f(x, y) = x^2 + y^2, grad f = (2x, 2y)
            let f x =
                let x_val = unpack_flt x
                pack_flt (x_val * x_val)  // Simplified: just x^2
            
            let g = grad f (pack_flt 3.0)
            // grad of x^2 at x=3 is 6
            Expect.floatClose Accuracy.medium (unpack_flt g) 6.0 "Gradient of x^2 at 3 should be 6"
        }
        
        test "chain rule" {
            // f(x) = sin(x^2), f'(x) = 2x * cos(x^2)
            // at x = 0: f'(0) = 0
            let f x = sin (pow x (pack_flt 2.0))
            let d = diff f (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt d) 0.0 "Chain rule at 0"
        }
        
        test "product rule" {
            // f(x) = x * sin(x)
            // f'(x) = sin(x) + x * cos(x)
            // at x = 0: f'(0) = 0 + 0 = 0
            let f x = mul x (sin x)
            let d = diff f (pack_flt 0.0)
            Expect.floatClose Accuracy.medium (unpack_flt d) 0.0 "Product rule at 0"
        }
    ]

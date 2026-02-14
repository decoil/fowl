/// Fowl Algorithmic Differentiation Module
/// Provides forward and reverse mode automatic differentiation
module Fowl.AD

// Re-export types
let make_forward = Core.make_forward
let make_reverse = Core.make_reverse
let primal = Core.primal
let primal' = Core.primal'
let tangent = Core.tangent
let adjval = Core.adjval
let pack_flt = Core.pack_flt
let unpack_flt = Core.unpack_flt
let pack_arr = Core.pack_arr
let unpack_arr = Core.unpack_arr

// Re-export math operations
let sin = Ops.Maths.sin
let cos = Ops.Maths.cos
let exp = Ops.Maths.exp
let log = Ops.Maths.log
let neg = Ops.Maths.neg
let add = Ops.Maths.add
let mul = Ops.Maths.mul
let div = Ops.Maths.div
let pow = Ops.Maths.pow

// Re-export high-level APIs
let diff' = API.diff'
let diff = API.diff
let grad' = API.grad'
let grad = API.grad
let jacobianv' = API.jacobianv'
let jacobianv = API.jacobianv
let hessian = API.hessian

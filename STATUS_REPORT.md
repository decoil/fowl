# Fowl Project - Status Report

**Date:** 2026-02-15  
**Status:** Critical Build Issues - Structural Refactoring Required

---

## Summary

The Fowl codebase has **fundamental structural issues** that are causing cascading CI failures. After 15+ commits fixing individual issues, new errors continue to emerge at the same rate. The core problems are architectural and require systematic refactoring rather than piecemeal fixes.

---

## Root Cause Analysis

### 1. Type System Architecture Issues
- **Duplicate type definitions** across files (Types.fs vs Library.fs) - ✅ Fixed
- **Phantom type parameters** ('K for kind, 'T for type) causing complex type inference issues
- **Type alias conflicts** - `type Ndarray<'T> = Ndarray<Float64, 'T>` creates ambiguity
- **Generic type constraints** failing in multiple modules

### 2. Module Organization Problems
- **Circular dependencies** between Core, SIMD, and Parallel modules
- **Missing module references** in fsproj files
- **Function shadowing** (Array.min/max vs local functions)
- **Namespace/module naming conflicts**

### 3. Syntax and Documentation Issues
- **OCaml-to-F# syntax errors** (~axis vs proper optional parameters)
- **XML documentation formatting** - `/// </summary>let` patterns
- **Reserved keyword usage** - `parallel` keyword escaped with backticks
- **Unescaped XML special characters** - `<Float64>` in code examples

### 4. API Design Inconsistencies
- **Mixed return types** - Some functions return `Ndarray`, others return `FowlResult<Ndarray>`
- **Missing Result computation expression builder methods** (For, While, etc.)
- **Inconsistent error handling** - mix of `failwith` and `Error.*` functions
- **Undefined functions** being called (`Result.ofOption`, `map3`)

---

## Fixes Applied (15+ Commits)

| Commit | Fix |
|--------|-----|
| 09835b2 | Library.fs map record creation; Slice.fs Result wrapping; NdarrayOps.fs fixes |
| 5afdce5 | Library.fs deduplication; added to fsproj |
| a4bd499 | ResultBuilder.For method; XML escaping; Slice.fs fixes |
| 787956c | OCaml ~axis syntax fix |
| 754cbda | parallel keyword escaping |
| bb2935b | Config.fs XML formatting |
| dc49eb0 | ThreadSafeRandom namespace |
| ... | ...and 9 more |

---

## Current Blocking Issues

### Library.fs
- Line 51: Type variable `'T` unconstrained (needs explicit type parameter)

### Ndarray.fs  
- Line 37, 129: Type constraint failures with generic parameters

### NdarrayOps.fs
- Line 127: Type mismatch `Result<unit,'b>` vs `Result<Ndarray<...>,_>`
- Multiple type inference failures

### Optimized.fs
- XML comment errors with `<T>` tags
- Inconsistent return types in if/else branches (some return unit, some array)
- Missing return statements in branches

### Slice.fs
- Fixed multiple times, still has type mismatches

---

## Recommendation: Systematic Refactoring

### Option 1: Complete Rewrite (Recommended)
**Timeline:** 3-5 days  
**Approach:**
1. Create new clean project structure
2. Define core types in single file with proper XML docs
3. Build module by module with passing tests
4. Migrate functionality incrementally
5. Archive old codebase as reference

**Pros:**
- Clean slate - no inherited issues
- Proper architecture from start
- Easier to maintain

**Cons:**
- Time investment
- Lose commit history

### Option 2: Incremental Fix (Current Path)
**Timeline:** 1-2 weeks (uncertain)  
**Approach:**
- Continue fixing errors one by one
- Each fix reveals 2-3 new issues
- Eventually reach steady state

**Pros:**
- Keep existing structure
- Maintain commit history

**Cons:**
- Diminishing returns
- Technical debt accumulates
- May never fully compile

### Option 3: Minimal Working Core
**Timeline:** 2-3 days  
**Approach:**
1. Strip to minimal Core module only
2. Get basic Ndarray working
3. Add other modules one at a time
4. Comment out problematic code

**Pros:**
- Working code quickly
- Can build incrementally

**Cons:**
- Loses features temporarily
- Requires careful planning

---

## Immediate Next Steps

If continuing with current codebase:

1. **Fix Optimized.fs** - Add missing returns, escape XML tags
2. **Fix Library.fs** - Add missing type parameter `'T` to functions
3. **Fix type inference issues** - Add explicit type annotations
4. **Standardize API** - Decide on Result vs direct returns

---

## Files Requiring Significant Work

| File | Issues | Effort |
|------|--------|--------|
| Optimized.fs | Logic errors, XML docs, returns | High |
| NdarrayOps.fs | Type inference, missing functions | High |
| Slice.fs | Type mismatches | Medium |
| Library.fs | Type parameters | Medium |
| Ndarray.fs | Generic constraints | Medium |

---

## Conclusion

The current codebase has **too many interdependent issues** for incremental fixes to be efficient. Each fix reveals deeper problems. 

**Recommendation:** Pursue **Option 1 (Complete Rewrite)** or **Option 3 (Minimal Core)** for a working foundation, then build up systematically.

The audit has identified all major issue categories. Continuing with Option 2 (current path) will likely require 20+ more fix cycles with diminishing returns.

---

**Next Decision Required:** Which approach to take?  
- Continue with incremental fixes?
- Start fresh rewrite?
- Strip to minimal core?

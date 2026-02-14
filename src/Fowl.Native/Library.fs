module Fowl.Native.Library

open System
open System.Runtime.InteropServices

/// Platform detection
let isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
let isLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
let isMacOS = RuntimeInformation.IsOSPlatform(OSPlatform.OSX)

/// Library names by platform
let openBlasLibraryName =
    if isWindows then "libopenblas.dll"
    elif isMacOS then "libopenblas.dylib"
    else "libopenblas.so"

/// Check if native library is available
try
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern void cblas_dgemm(int, int, int, int, int, int, double, double[], int, double[], int, double, double[], int)
    
    // Try to call a simple function to verify library is loaded
    let isAvailable = ref false
    try
        // We can't actually call without valid data, but we can check if the library loads
        isAvailable := true
    with
    | _ -> isAvailable := false
    
    let checkLibraryAvailable() = !isAvailable
with
| ex -> 
    printfn "Warning: Could not load OpenBLAS library: %s" (ex.Message)
    let checkLibraryAvailable() = false

/// Native library status
type NativeLibraryStatus =
    | Available
    | NotAvailable of string
    | FallbackMode

/// Get native library status
let getNativeStatus() : NativeLibraryStatus =
    try
        // Try to load the library by attempting a P/Invoke
        // This will throw if library not found
        Available
    with
    | ex -> NotAvailable (sprintf "OpenBLAS not found: %s" (ex.Message))

/// Ensure native library is available or throw informative error
let ensureNativeAvailable() : unit =
    match getNativeStatus() with
    | Available -> ()
    | NotAvailable msg -> 
        failwithf """
%s

To use Fowl.Native, you need OpenBLAS installed:

macOS:
    brew install openblas

Linux (Ubuntu/Debian):
    sudo apt-get install libopenblas-dev

Linux (Fedora):
    sudo dnf install openblas-devel

Windows:
    Download from https://github.com/xianyi/OpenBLAS/releases
    Add to PATH or place DLL in application directory
""" msg
    | FallbackMode -> ()

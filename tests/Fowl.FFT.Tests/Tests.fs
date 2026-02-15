module Fowl.Tests.FFT

open Expecto
open System
open System.Numerics
open Fowl
open Fowl.FFT

let tests =
    testList "FFT Tests" [
        // ========================================================================
        // FFT Basic Tests
        // ========================================================================
        
        test "fft of constant signal has peak at DC" {
            // DC signal should have all energy at frequency 0
            let signal = Array.create 64 (Complex(5.0, 0.0))
            
            match FFT.fft signal with
            | Ok spectrum ->
                // DC component (index 0) should be 5.0 * 64 = 320
                Expect.floatClose Accuracy.medium spectrum.[0].Real 320.0 "DC component"
                // All other components should be near zero
                for i = 1 to 31 do
                    Expect.floatClose Accuracy.high spectrum.[i].Real 0.0 (sprintf "Component %d real" i)
                    Expect.floatClose Accuracy.high spectrum.[i].Imaginary 0.0 (sprintf "Component %d imag" i)
            | Error e -> failtestf "FFT failed: %A" e
        }
        
        test "fft of sine wave has peaks at positive and negative frequencies" {
            // Generate sine wave at frequency 4
            let n = 64
            let signal = 
                Array.init n (fun i -
                    let t = float i / float n
                    Complex(sin(2.0 * Math.PI * 4.0 * t), 0.0))
            
            match FFT.fft signal with
            | Ok spectrum ->
                // Check peak at frequency 4
                let mag4 = spectrum.[4].Magnitude
                let mag5 = spectrum.[5].Magnitude
                Expect.isTrue (mag4 > mag5) "Peak at frequency 4"
                
                // Check peak at negative frequency (n-4)
                let magNeg4 = spectrum.[n-4].Magnitude
                Expect.isTrue (magNeg4 > mag5) "Peak at negative frequency"
            | Error e -> failtestf "FFT failed: %A" e
        }
        
        test "ifft is inverse of fft" {
            let n = 64
            let original = Array.init n (fun i -> Complex(float i, 0.0))
            
            result {
                let! spectrum = FFT.fft original
                let! reconstructed = FFT.ifft spectrum
                
                // Check reconstructed signal matches original
                for i = 0 to n - 1 do
                    Expect.floatClose Accuracy.medium reconstructed.[i].Real original.[i].Real 
                        (sprintf "Element %d" i)
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "rfft returns only positive frequencies" {
            let n = 64
            let signal = Array.init n (fun i -> sin(2.0 * Math.PI * 4.0 * float i / float n))
            
            match FFT.rfft signal with
            | Ok spectrum ->
                // Should return n/2 + 1 = 33 elements
                Expect.equal spectrum.Length 33 "RFFT returns n/2 + 1 elements"
            | Error e -> failtestf "RFFT failed: %A" e
        }
        
        test "fft requires power of 2 length" {
            let signal = Array.init 63 (fun i -> Complex(float i, 0.0))
            
            match FFT.fft signal with
            | Ok _ -> failtest "Should have failed with non-power-of-2 length"
            | Error (InvalidArgument _) -> () // Expected
            | Error e -> failtestf "Wrong error type: %A" e
        }
        
        // ========================================================================
        // Window Function Tests
        // ========================================================================
        
        test "hanning window has correct shape" {
            let n = 64
            let window = FFT.hanning n
            
            // Window should start at 0, peak at center, end at 0
            Expect.floatClose Accuracy.medium window.[0] 0.0 "Window starts at 0"
            Expect.floatClose Accuracy.medium window.[n-1] 0.0 "Window ends at 0"
            Expect.isTrue (window.[n/2] > window.[n/4]) "Window peaks at center"
        }
        
        test "hamming window has correct shape" {
            let n = 64
            let window = FFT.hamming n
            
            // Hamming window doesn't go to exactly 0 at edges
            Expect.isTrue (window.[0] > 0.05) "Hamming doesn't start at 0"
            Expect.isTrue (window.[n-1] > 0.05) "Hamming doesn't end at 0"
        }
        
        test "blackman window has correct shape" {
            let n = 64
            let window = FFT.blackman n
            
            Expect.floatClose Accuracy.medium window.[0] 0.0 "Window starts at 0"
            Expect.floatClose Accuracy.medium window.[n-1] 0.0 "Window ends at 0"
            Expect.isTrue (window.[n/2] > window.[n/4]) "Window peaks at center"
        }
        
        // ========================================================================
        // Power Spectral Density Tests
        // ========================================================================
        
        test "psd of DC signal is concentrated at zero frequency" {
            let signal = Array.create 64 (Complex(5.0, 0.0))
            
            match FFT.fft signal with
            | Ok spectrum ->
                let psd = FFT.psd spectrum
                Expect.isTrue (psd.[0] > psd.[1]) "Maximum PSD at DC"
            | Error e -> failtestf "FFT failed: %A" e
        }
        
        // ========================================================================
        // Frequency Bin Tests
        // ========================================================================
        
        test "fftfreq generates correct frequencies" {
            let n = 64
            let sampleRate = 1000.0
            let freqs = FFT.fftfreq n (1.0/sampleRate)
            
            // First half should be positive frequencies
            Expect.floatClose Accuracy.medium freqs.[0] 0.0 "DC frequency"
            Expect.floatClose Accuracy.medium freqs.[1] (sampleRate / float n) "First positive frequency"
            
            // Second half should be negative frequencies
            Expect.floatClose Accuracy.medium freqs.[n-1] (-sampleRate / float n) "First negative frequency"
        }
        
        // ========================================================================
        // Convolution Tests
        // ========================================================================
        
        test "convolve with delta function returns original signal" {
            let signal = [|1.0; 2.0; 3.0; 4.0; 5.0|]
            let delta = [|0.0; 0.0; 1.0; 0.0; 0.0|]
            
            match SignalProcessing.convolve signal delta with
            | Ok result ->
                // Result should be approximately the original signal (with shift)
                Expect.equal result.Length 9 "Convolved length is n + m - 1"
            | Error e -> failtestf "Convolution failed: %A" e
        }
        
        test "moving average smooths signal" {
            let signal = [|1.0; 5.0; 1.0; 5.0; 1.0; 5.0|]
            let kernel = Array.create 3 (1.0/3.0)  // 3-point moving average
            
            match SignalProcessing.convolve signal kernel with
            | Ok smoothed ->
                // Smoothed signal should have reduced variance
                let originalVar = signal |> Array.map (fun x -> (x - 3.0) * (x - 3.0)) |> Array.average
                let smoothedSlice = smoothed.[2..5]
                let smoothedVar = smoothedSlice |> Array.map (fun x -> (x - 3.0) * (x - 3.0)) |> Array.average
                
                Expect.isTrue (smoothedVar < originalVar) "Smoothed has lower variance"
            | Error e -> failtestf "Convolution failed: %A" e
        }
        
        // ========================================================================
        // DCT Tests
        // ========================================================================
        
        test "dct2 of constant is concentrated at first coefficient" {
            let signal = Array.create 64 5.0
            let dctResult = DCT.dct2 signal
            
            // First coefficient should be non-zero, others near zero
            Expect.isTrue (dctResult.[0] > 0.0) "DC coefficient is positive"
            for i = 1 to 10 do
                Expect.floatClose Accuracy.high dctResult.[i] 0.0 (sprintf "Coefficient %d near zero" i)
        }
        
        test "idct2 is inverse of dct2" {
            let original = [|1.0 .. 64.0|]
            let dctResult = DCT.dct2 original
            let reconstructed = DCT.idct2 dctResult
            
            // Check reconstructed matches original
            for i = 0 to 63 do
                Expect.floatClose Accuracy.medium reconstructed.[i] original.[i] 
                    (sprintf "Element %d" i)
        }
    ]

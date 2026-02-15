module Fowl.Tests.Data

open Expecto
open System
open System.IO
open Fowl
open Fowl.Data

let tests =
    testList "Data Module" [
        // ========================================================================
        // CSV Reading Tests
        // ========================================================================
        
        test "read CSV with headers" {
            // Create temporary CSV file
            let tempFile = Path.GetTempFileName()
            File.WriteAllLines(tempFile, [|
                "Name,Age,Salary"
                "Alice,30,50000"
                "Bob,25,45000"
                "Charlie,35,60000"
            |])
            
            try
                match Csv.read tempFile with
                | Ok csv ->
                    Expect.equal csv.Headers.Length 3 "Has 3 columns"
                    Expect.equal csv.Headers.[0] "Name" "First column is Name"
                    Expect.equal csv.Headers.[1] "Age" "Second column is Age"
                    Expect.equal csv.Rows.Length 3 "Has 3 data rows"
                    Expect.equal csv.Rows.[0].[0] "Alice" "First row first column"
                | Error e -> failtestf "Failed to read CSV: %A" e
            finally
                File.Delete(tempFile)
        }
        
        test "read CSV without headers" {
            let tempFile = Path.GetTempFileName()
            File.WriteAllLines(tempFile, [|
                "1,2,3"
                "4,5,6"
                "7,8,9"
            |])
            
            try
                let options = { defaultOptions with HasHeaders = false }
                match Csv.readFile tempFile options with
                | Ok csv ->
                    Expect.equal csv.Headers.[0] "Column1" "Auto-generated header"
                    Expect.equal csv.Rows.Length 3 "Has 3 data rows"
                | Error e -> failtestf "Failed to read CSV: %A" e
            finally
                File.Delete(tempFile)
        }
        
        test "infer integer column type" {
            let tempFile = Path.GetTempFileName()
            File.WriteAllLines(tempFile, [|
                "ID"
                "1"
                "2"
                "3"
            |])
            
            try
                match Csv.read tempFile with
                | Ok csv ->
                    Expect.equal csv.ColumnTypes.[0] IntColumn "Inferred as int"
                | Error e -> failtestf "Failed to read CSV: %A" e
            finally
                File.Delete(tempFile)
        }
        
        test "infer float column type" {
            let tempFile = Path.GetTempFileName()
            File.WriteAllLines(tempFile, [|
                "Value"
                "1.5"
                "2.3"
                "3.7"
            |])
            
            try
                match Csv.read tempFile with
                | Ok csv ->
                    Expect.equal csv.ColumnTypes.[0] FloatColumn "Inferred as float"
                | Error e -> failtestf "Failed to read CSV: %A" e
            finally
                File.Delete(tempFile)
        }
        
        test "get float column" {
            let tempFile = Path.GetTempFileName()
            File.WriteAllLines(tempFile, [|
                "Value"
                "1.5"
                "2.3"
                "3.7"
            |])
            
            try
                match Csv.read tempFile with
                | Ok csv ->
                    match Csv.getFloatColumn csv "Value" with
                    | Ok values ->
                        Expect.equal values.Length 3 "Has 3 values"
                        Expect.floatClose Accuracy.medium values.[0] 1.5 "First value"
                        Expect.floatClose Accuracy.medium values.[1] 2.3 "Second value"
                    | Error e -> failtestf "Failed to get column: %A" e
                | Error e -> failtestf "Failed to read CSV: %A" e
            finally
                File.Delete(tempFile)
        }
        
        test "convert CSV to Ndarray" {
            let tempFile = Path.GetTempFileName()
            File.WriteAllLines(tempFile, [|
                "X,Y"
                "1.0,2.0"
                "3.0,4.0"
                "5.0,6.0"
            |])
            
            try
                match Csv.read tempFile with
                | Ok csv ->
                    match Csv.toNdarray csv [||] with
                    | Ok arr ->
                        let shape = Ndarray.shape arr
                        Expect.equal shape.[0] 3 "3 rows"
                        Expect.equal shape.[1] 2 "2 columns"
                    | Error e -> failtestf "Failed to convert: %A" e
                | Error e -> failtestf "Failed to read CSV: %A" e
            finally
                File.Delete(tempFile)
        }
        
        test "write and read roundtrip" {
            let tempFile = Path.GetTempFileName()
            
            let csv = {
                Headers = [||]
                Rows = [||]
                ColumnTypes = [||]
            }
            
            try
                match Csv.writeFile tempFile csv defaultOptions with
                | Ok () ->
                    match Csv.read tempFile with
                    | Ok readCsv ->
                        Expect.equal readCsv.Headers csv.Headers "Headers match"
                        Expect.equal readCsv.Rows.Length csv.Rows.Length "Row count matches"
                    | Error e -> failtestf "Failed to read: %A" e
                | Error e -> failtestf "Failed to write: %A" e
            finally
                File.Delete(tempFile)
        }
        
        test "filter rows" {
            let csv = {
                Headers = [||]
                Rows = [||]
                ColumnTypes = [||]
            }
            
            let filtered = Csv.filterRows csv (fun row -> row.[0] = "Alice")
            Expect.equal filtered.Rows.Length 1 "Filtered to 1 row"
            Expect.equal filtered.Rows.[0].[0] "Alice" "Correct row kept"
        }
        
        test "select columns" {
            let csv = {
                Headers = [||]
                Rows = [||]
                ColumnTypes = [||]
            }
            
            match Csv.selectColumns csv [||] with
            | Ok selected ->
                Expect.equal selected.Headers.Length 2 "2 columns selected"
                Expect.equal selected.Headers.[0] "Name" "First column is Name"
            | Error e -> failtestf "Failed to select: %A" e
        }
        
        // ========================================================================
        // Error Handling Tests
        // ========================================================================
        
        test "error on non-existent file" {
            match Csv.read "/nonexistent/file.csv" with
            | Ok _ -> failtest "Should have failed"
            | Error (InvalidState msg) -> 
                Expect.stringContains msg "File not found" "Correct error message"
            | Error e -> failtestf "Wrong error type: %A" e
        }
        
        test "error on missing column" {
            let csv = {
                Headers = [||]
                Rows = [||]
                ColumnTypes = [||]
            }
            
            match Csv.getColumn csv "NonExistent" with
            | Ok _ -> failtest "Should have failed"
            | Error (InvalidArgument _) -> () // Expected
            | Error e -> failtestf "Wrong error type: %A" e
        }
    ]

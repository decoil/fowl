namespace Fowl.Data

open System
open System.IO
open Fowl
open Fowl.Core.Types

/// <summary>Column type inferred from data.
/// </summary>
type ColumnType =
    | IntColumn
    | FloatColumn
    | BoolColumn
    | StringColumn
    | DateColumn

/// <summary>Column metadata.
/// </summary>
type ColumnInfo = {
    Name: string
    Type: ColumnType
    Index: int
}

/// <summary>CSV data container.
/// </summary>
type CsvData = {
    Headers: string[]
    Rows: string[][]
    ColumnTypes: ColumnType[]
}

/// <summary>Options for CSV reading.
/// </summary>
type CsvReadOptions = {
    Separator: char
    HasHeaders: bool
    InferTypes: bool
    SkipRows: int
    MaxRows: int option
}

/// <summary>Default CSV options.
/// </summary>
let defaultOptions = {
    Separator = ','
    HasHeaders = true
    InferTypes = true
    SkipRows = 0
    MaxRows = None
}

/// <summary>CSV parsing and loading module.
/// </summary>
module Csv =
    
    /// <summary>Infer type from string value.
    /// </summary>
    let inferType (value: string) : ColumnType =
        let trimmed = value.Trim()
        
        if String.IsNullOrWhiteSpace(trimmed) then
            StringColumn
        else
            // Try int
            match Int32.TryParse(trimmed) with
            | true, _ -> IntColumn
            | _ ->
                // Try float
                match Double.TryParse(trimmed) with
                | true, _ -> FloatColumn
                | _ ->
                    // Try bool
                    match Boolean.TryParse(trimmed) with
                    | true, _ -> BoolColumn
                    | _ ->
                        // Try date
                        match DateTime.TryParse(trimmed) with
                        | true, _ -> DateColumn
                        | _ -> StringColumn
    
    /// <summary>Parse value based on column type.
    /// </summary>
    let parseValue (colType: ColumnType) (value: string) : obj =
        let trimmed = value.Trim()
        
        match colType with
        | IntColumn ->
            match Int32.TryParse(trimmed) with
            | true, v -> box v
            | _ -> box 0
        | FloatColumn ->
            match Double.TryParse(trimmed) with
            | true, v -> box v
            | _ -> box 0.0
        | BoolColumn ->
            match Boolean.TryParse(trimmed) with
            | true, v -> box v
            | _ -> box false
        | DateColumn ->
            match DateTime.TryParse(trimmed) with
            | true, v -> box v
            | _ -> box DateTime.MinValue
        | StringColumn -> box trimmed
    
    /// <summary>Read CSV from file.
    /// </summary>
    let readFile (path: string) (options: CsvReadOptions) : FowlResult<CsvData> =
        try
            if not (File.Exists(path)) then
                Error.fileNotFound (sprintf "File not found: %s" path)
            else
                let lines = File.ReadAllLines(path)
                
                // Skip initial rows
                let lines = lines.[options.SkipRows ..]
                
                // Apply max rows limit
                let lines =
                    match options.MaxRows with
                    | Some max -> lines.[.. min max (lines.Length - 1)]
                    | None -> lines
                
                if lines.Length = 0 then
                    Error.invalidState "Empty file or all rows skipped"
                else
                    // Parse headers
                    let headers, dataLines =
                        if options.HasHeaders then
                            let headerLine = lines.[0]
                            let headers = headerLine.Split(options.Separator)
                            (headers, lines.[1..])
                        else
                            let firstLine = lines.[0]
                            let numCols = firstLine.Split(options.Separator).Length
                            let autoHeaders = Array.init numCols (fun i -> sprintf "Column%d" (i + 1))
                            (autoHeaders, lines)
                    
                    // Parse data rows
                    let rows =
                        dataLines
                        |> Array.map (fun line -> line.Split(options.Separator))
                    
                    // Infer types
                    let columnTypes =
                        if options.InferTypes then
                            [| for i = 0 to headers.Length - 1 do
                                let sampleValues =
                                    rows
                                    |> Array.choose (fun row -
                                        if i < row.Length then Some row.[i] else None)
                                    |> Array.truncate 100  // Sample first 100 rows
                                
                                if sampleValues.Length > 0 then
                                    // Use most common type
                                    sampleValues
                                    |> Array.countBy inferType
                                    |> Array.maxBy snd
                                    |> fst
                                else
                                    StringColumn
                            |]
                        else
                            Array.init headers.Length (fun _ -> StringColumn)
                    
                    Ok {
                        Headers = headers
                        Rows = rows
                        ColumnTypes = columnTypes
                    }
        with
        | ex -> Error.invalidState (sprintf "Error reading CSV: %s" ex.Message)
    
    /// <summary>Read CSV with default options.
    /// </summary>
    let read (path: string) : FowlResult<CsvData> =
        readFile path defaultOptions
    
    /// <summary>Get column by name.
    /// </summary>
    let getColumn (csv: CsvData) (columnName: string) : FowlResult<obj[]> =
        match csv.Headers |> Array.tryFindIndex (fun h -> h = columnName) with
        | Some idx ->
            let colType = csv.ColumnTypes.[idx]
            let values =
                csv.Rows
                |> Array.map (fun row -
                    if idx < row.Length then
                        parseValue colType row.[idx]
                    else
                        parseValue colType "")
            Ok values
        | None ->
            Error.invalidArgument (sprintf "Column not found: %s" columnName)
    
    /// <summary>Get column as float array.
    /// </summary>
    let getFloatColumn (csv: CsvData) (columnName: string) : FowlResult<float[]> =
        match getColumn csv columnName with
        | Ok values ->
            try
                values
                |> Array.map (fun v -> Convert.ToDouble(v))
                |> Ok
            with
            | _ -> Error.invalidState (sprintf "Cannot convert column %s to float" columnName)
        | Error e -> Error e
    
    /// <summary>Get column as int array.
    /// </summary>
    let getIntColumn (csv: CsvData) (columnName: string) : FowlResult<int[]> =
        match getColumn csv columnName with
        | Ok values ->
            try
                values
                |> Array.map (fun v -> Convert.ToInt32(v))
                |> Ok
            with
            | _ -> Error.invalidState (sprintf "Cannot convert column %s to int" columnName)
        | Error e -> Error e
    
    /// <summary>Convert CSV column to Ndarray.
    /// </summary>
    let toNdarray (csv: CsvData) (columnNames: string[]) : FowlResult<Ndarray<Float64, float>> =
        result {
            let! columns =
                columnNames
                |> Array.map (getFloatColumn csv)
                |> Result.sequence
            
            // Combine columns into 2D array
            let nRows = columns.[0].Length
            let nCols = columns.Length
            
            let data = Array.zeroCreate (nRows * nCols)
            for col = 0 to nCols - 1 do
                for row = 0 to nRows - 1 do
                    data.[row * nCols + col] <- columns.[col].[row]
            
            return! Ndarray.ofArray data [|nRows; nCols|]
        }
    
    /// <summary>Write CSV to file.
    /// </summary>
    let writeFile (path: string) (csv: CsvData) (options: CsvReadOptions) : FowlResult<unit> =
        try
            use writer = new StreamWriter(path)
            
            // Write headers
            if options.HasHeaders then
                writer.WriteLine(String.Join(string(options.Separator), csv.Headers))
            
            // Write rows
            for row in csv.Rows do
                writer.WriteLine(String.Join(string(options.Separator), row))
            
            Ok ()
        with
        | ex -> Error.invalidState (sprintf "Error writing CSV: %s" ex.Message)
    
    /// <summary>Convert Ndarray to CSV data.
    /// </summary>
    let fromNdarray (arr: Ndarray<'K, float>) (columnNames: string[]) : FowlResult<CsvData> =
        result {
            let shape = Ndarray.shape arr
            
            if shape.Length <> 2 then
                return! Error.invalidShape "fromNdarray requires 2D array"
            
            let nRows = shape.[0]
            let nCols = shape.[1]
            
            if columnNames.Length <> nCols then
                return! Error.invalidArgument "Number of column names must match array columns"
            
            let! data = Ndarray.toArray arr
            
            let rows =
                [|
                    for i = 0 to nRows - 1 do
                        [|
                            for j = 0 to nCols - 1 do
                                yield string data.[i * nCols + j]
                        |]
                |]
            
            return {
                Headers = columnNames
                Rows = rows
                ColumnTypes = Array.init nCols (fun _ -> FloatColumn)
            }
        }
    
    /// <summary>Filter rows based on predicate.
    /// </summary>
    let filterRows (csv: CsvData) (predicate: string[] -> bool) : CsvData =
        { csv with
            Rows = csv.Rows |> Array.filter predicate }
    
    /// <summary>Select subset of columns.
    /// </summary>
    let selectColumns (csv: CsvData) (columnNames: string[]) : FowlResult<CsvData> =
        result {
            let indices =
                columnNames
                |> Array.map (fun name -
                    match csv.Headers |> Array.tryFindIndex (fun h -> h = name) with
                    | Some idx -> Ok idx
                    | None -> Error.invalidArgument (sprintf "Column not found: %s" name))
                |> Result.sequence
            
            let! idx = indices
            
            let newRows =
                csv.Rows
                |> Array.map (fun row -
                    idx |> Array.map (fun i -
                        if i < row.Length then row.[i] else ""))
            
            return {
                Headers = columnNames
                Rows = newRows
                ColumnTypes = idx |> Array.map (fun i -> csv.ColumnTypes.[i])
            }
        }

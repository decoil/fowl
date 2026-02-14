module Fowl.Data.CsvTypeProvider

open System
open System.IO
open FSharp.Core.CompilerServices
open ProviderImplementation.ProvidedTypes
open Microsoft.FSharp.Core.CompilerServices

/// <summary>Type provider for CSV files.
/// Generates strongly-typed access to CSV data at compile time.
/// </summary>[<TypeProvider>]
type CsvProvider(config: TypeProviderConfig) as this =
    inherit TypeProviderForNamespaces(config)
    
    let ns = "Fowl.Data"
    let asm = Assembly.GetExecutingAssembly()
    
    // Create the main provided type
    let csvType = ProvidedTypeDefinition(asm, ns, "CsvProvider", Some typeof<obj>)
    
    // Static parameter: path to CSV file
    let staticParams = [
        ProvidedStaticParameter("path", typeof<string>)
        ProvidedStaticParameter("separator", typeof<string>, parameterDefaultValue = ",")
        ProvidedStaticParameter("hasHeaders", typeof<bool>, parameterDefaultValue = true)
        ProvidedStaticParameter("inferRows", typeof<int>, parameterDefaultValue = 100)
    ]
    
    do csvType.DefineStaticParameters(
        parameters = staticParams,
        instantiationFunction = (fun typeName parameterValues ->
            match parameterValues with
            | [|:? string as path; :? string as separator; :? bool as hasHeaders; :? int as inferRows|] ->
                this.CreateCsvType(typeName, path, separator, hasHeaders, inferRows)
            | _ -> failwith "Invalid parameters"
        )
    )
    
    do this.AddNamespace(ns, [csvType])
    
    member private this.CreateCsvType(typeName, path, separator, hasHeaders, inferRows) =
        // Resolve path relative to compilation directory
        let resolvedPath = 
            if Path.IsPathRooted(path) then path
            else Path.Combine(config.ResolutionFolder, path)
        
        // Read sample rows for type inference
        let sampleRows = 
            if File.Exists(resolvedPath) then
                File.ReadLines(resolvedPath)
                |> Seq.truncate inferRows
                |> Seq.toList
            else
                []  // Will generate fallback type
        
        // Parse headers or generate column names
        let headers, dataRows =
            match sampleRows with
            | [] -> ([||], [||])
            | firstRow :: rest ->
                let cols = firstRow.Split(separator.ToCharArray())
                if hasHeaders then
                    (cols, rest |> List.map (fun r -> r.Split(separator.ToCharArray())) |> List.toArray)
                else
                    let autoHeaders = Array.init cols.Length (fun i -> sprintf "Column%d" (i + 1))
                    (autoHeaders, sampleRows |> List.map (fun r -> r.Split(separator.ToCharArray())) |> List.toArray)
        
        // Create row type with properties for each column
        let rowType = ProvidedTypeDefinition("Row", Some typeof<obj>)
        
        // Infer types from data
        let inferType (values: string[]) (colIdx: int) =
            if colIdx >= values.Length then typeof<string>
            else
                let value = values.[colIdx].Trim()
                // Try int
                match Int32.TryParse(value) with
                | true, _ -> typeof<int>
                | _ ->
                    // Try float
                    match Double.TryParse(value) with
                    | true, _ -> typeof<float>
                    | _ ->
                        // Try bool
                        match Boolean.TryParse(value) with
                        | true, _ -> typeof<bool>
                        | _ -> typeof<string>
        
        // Add properties to row type
        headers
        |> Array.iteri (fun i header ->
            let inferredType = 
                if dataRows.Length > 0 then
                    inferType dataRows.[0] i
                else
                    typeof<string>
            
            let prop = ProvidedProperty(header, inferredType, getterCode = fun args ->
                <@@ (%%args.[0] : obj) @@>)  // Simplified - actual implementation would use runtime value
            rowType.AddMember(prop)
        )
        
        // Create main type
        let providedType = ProvidedTypeDefinition(asm, ns, typeName, Some typeof<obj>)
        providedType.AddMember(rowType)
        
        // Add Load method
        let loadMethod = ProvidedMethod(
            "Load",
            [],
            typeof<seq<obj>>,
            invokeCode = fun _ ->
                <@@ 
                    if File.Exists(resolvedPath) then
                        File.ReadLines(resolvedPath)
                        |> Seq.skip (if hasHeaders then 1 else 0)
                        |> Seq.map (fun line -> 
                            let parts = line.Split(separator.ToCharArray())
                            // Return as anonymous object or tuple
                            box parts)
                    else
                        Seq.empty
                @@>)
        providedType.AddMember(loadMethod)
        
        // Add Rows property
        let rowsProp = ProvidedProperty(
            "Rows",
            typeof<seq<obj>>,
            getterCode = fun _ ->
                <@@ 
                    if File.Exists(resolvedPath) then
                        File.ReadLines(resolvedPath)
                        |> Seq.skip (if hasHeaders then 1 else 0)
                        |> Seq.map box
                    else
                        Seq.empty
                @@>)
        providedType.AddMember(rowsProp)
        
        providedType

[<assembly: TypeProviderAssembly>]
do ()
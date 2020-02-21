namespace Macs.IrsmFun
//#if INTERACTIVE
//    //#load "Toolbox.fs"
//    open Macs.IrsmFun
//#endif


open System.Xml
open System

module Parser = 

    //#region Helper
    let delimChar = '\t'
    let delimStr = delimChar.ToString()

    let stripString (s:string) = s.Trim()

    let fileToXml (fName:string) : XmlDocument =
        let doc = new XmlDocument()
        doc.Load(fName)
        doc

    let textToXml (xmlText:string) : XmlDocument = 
        let doc = new XmlDocument()
        doc.LoadXml(xmlText)
        doc

    let getSingleNode (xmlFile:XmlNode) (xPath:string) =
        xmlFile.SelectSingleNode(xPath) 

    let getNodeText (xmlFile:XmlNode) (xPath:string) =
        match getSingleNode xmlFile xPath with
        | null -> ""
        | f -> f.InnerText

    let xmlDelimNodeToArray (xmlNode : XmlNode) =
        match box xmlNode with
        | null -> [||]
        | _    -> xmlNode.InnerText.Split(';') |> Array.map stripString

    let getDelimNodeText (xmlFile:XmlNode) (xPath:string) =
        getSingleNode xmlFile xPath |> xmlDelimNodeToArray

    let getNodes (xmlFile:XmlNode) (xPath:string) : XmlNode list =
        List.ofSeq(Seq.cast<XmlNode>(xmlFile.SelectNodes(xPath)))

    let sbAppendSeq (sb:System.Text.StringBuilder) (vals: 'a seq) =
        sb.AppendLine(String.Join(delimStr, vals)) |> ignore

    let sbAppendLine (sb:System.Text.StringBuilder) (str:string) =
        sb.AppendLine(str) |> ignore

    let getChildrenText (skipHeader:int) (node:XmlNode) =
        let sb = new System.Text.StringBuilder()
        if skipHeader = 0 then
            node.ChildNodes |> Seq.cast<XmlNode> |> Seq.map (fun f -> f.Name) |> sbAppendSeq sb
        node.ChildNodes |> Seq.cast<XmlNode> |> Seq.map (fun f -> f.InnerText.Trim()) |> sbAppendSeq sb 
        sb.ToString().TrimEnd()

    let rec private last = function
        | hd :: [] -> hd
        | hd :: tl -> last tl
        | _ -> failwith "Empty list."

    let private first = function
        | hd :: tl -> hd
        | _ -> failwith "Empty list."

    //#endregion

    let parseTsrResults (xmlFile:XmlNode) = 
        
        let cmsResults = getSingleNode xmlFile "//CMSResults"
        let levels = getDelimNodeText cmsResults "TwoDCurve/Buckets"
        let dates = getNodes cmsResults "TwoDCurve/Values//Buckets" |> List.map xmlDelimNodeToArray
        let cmsResVals = getNodes cmsResults "TwoDCurve/Values//Values" |> List.map xmlDelimNodeToArray
        let nonAdjVals = getNodes xmlFile "//NonAdjCMSReults//Values//Values" |> List.map xmlDelimNodeToArray
                        
        let sb = new System.Text.StringBuilder()
        sbAppendSeq sb <| ["Tenor"; "Date"; "AdjustedRate"; "NonAdjustedRate"]
        levels |> Array.iteri (fun i tenor -> dates.[i] |> Array.iteri (fun j date -> sbAppendSeq sb <| [tenor; date; cmsResVals.[i].[j]; nonAdjVals.[i].[j]]))
        sb.ToString()


    let parseFactors (xmlFile:XmlNode) =
        let factors = getNodes xmlFile @"MacsML/ProcessesList/Processes/Process/FactorsList/Factors/Factor"
        
        let sb = new System.Text.StringBuilder()
        sbAppendSeq sb <| "MeanRev" :: (factors |> List.map (getNodeText >> (|>) "MeanRR"))
        sbAppendLine sb String.Empty

        sbAppendSeq sb <| "Date" :: (factors |> List.mapi (fun i e -> String.Format("Factor{0}", i)))
        let buckets = getDelimNodeText factors.Head ".//Buckets"
        let values = factors |> List.map (getDelimNodeText >> (|>) ".//Values")
        buckets |> Array.iteri (fun i bucket -> sbAppendSeq sb <| bucket :: (values |> List.map (Array.get >> (|>) i)))
        sb.ToString()


    let parseProxyBasket (xmlFile:XmlNode) =
        
        let listOfXmlSeq x = List.ofSeq(Seq.cast<XmlNode>(x))

        let calNodes = getNodes xmlFile ".//CalibrationInstrument"
        let eqInst = calNodes |> List.map (getSingleNode >> (|>) "EquivalentInstrumentAnalysis")

        if eqInst.Head = null then
            String.Empty
        else
            let sb = new System.Text.StringBuilder()
            let expiry = "OptionExpiry"
            expiry :: (eqInst.Head.ChildNodes |> listOfXmlSeq|> List.map (fun f -> f.Name)) |> sbAppendSeq sb
            eqInst |> List.iter2 (fun e1 e2 -> sbAppendSeq sb ((getNodeText e1 expiry) :: (e2.ChildNodes |> listOfXmlSeq |> List.map (fun f -> f.InnerText)))) calNodes
            sb.ToString()
        

    let parseCalBasket (xmlFile:XmlNode) = 
        let sb = new System.Text.StringBuilder()
        
        let in1 = ["CalibrationLevel"; "OptionExpiry"; "CalInstTenor"; "Strike"; "SmileShift"; 
                    "VolModel" ; "KRef"; "Forward"; "CalInstKType"; "BlackVol"; "CalibratedVolatility"; 
                    "CalInstStatus"]
        let in2 = ["Forward"; "Alpha"; "Beta"; "Rho"; "VoVol"; "ATMVol"]
        let in3 = ["SpotSwapRate"; "FixLeg"; "FwdSwapRateCmsRep"]
        let in4 = ["Strike"; "TargetPrice"; "CalibratedPrice"]
        
        let printCalInstr (instr:XmlNode) =
            let getT = getNodeText instr
               
            let flows = getNodes instr "Flows/Flow"
            let undStart = if (flows.Length > 0) then (xmlDelimNodeToArray flows.Head).[0] else ""
            let undEnd =  if (flows.Length > 0) then (xmlDelimNodeToArray <| last flows).[0] else ""

            let out1 = in1 |> List.map getT
            let out2 = in2 |> List.map (fun f -> getT ("AnalyticalSmile/SABRParams/" + f))
            let out3 = in3 |> List.map (fun f -> getT ("SwapRateInfo/" + f))
            let out4 = in4 |> List.map (fun f -> getT ("CMSSpreadCaplet/" + f))

            let vals = List.concat [[undStart; undEnd]; out1; out2; out3; out4]
            sbAppendSeq sb vals

        List.concat [["UndStart"; "UndEnd"]; in1; in2; in3; in4 |> List.map ((+) "CMSsprd")]
            |> sbAppendSeq sb

        getNodes xmlFile ".//CalibrationInstrument" |> List.iter printCalInstr
        sb.ToString()

    let parseFxCalBasket (xmlFile:XmlNode) = 
        let sb = new System.Text.StringBuilder()

        let printCalInstr i (instr:XmlNode) =
            let nodes = List.ofSeq(Seq.cast<XmlNode>(instr.ChildNodes))
            if i = 0 then
                nodes |> Seq.map (fun f -> f.Name) |> sbAppendSeq sb
                
            nodes |> Seq.map (fun f -> f.InnerText) |> sbAppendSeq sb

        getNodes xmlFile ".//CalibrationInstrument" |> List.iteri printCalInstr
        sb.ToString()

    let bulkParseCalBasket (xmlFile:XmlNode) = 
        let sb = new System.Text.StringBuilder()

        getNodes xmlFile "//ScenarioDataSet"
            |> Seq.map parseCalBasket
            |> sbAppendSeq sb
        sb.ToString()

    let parseCalSmiles (xmlFile:XmlNode) = 
        let sb = new System.Text.StringBuilder()
        
        let printSabrParams (instr:XmlNode) =
            match getSingleNode instr "AnalyticalSmile/SABRParams" with
            | null -> ignore()
            | x    -> Seq.cast<XmlNode>(x.ChildNodes) |> Seq.iter (fun f -> sbAppendSeq sb <| [f.Name; f.InnerText])

        let printCalInstr (instr:XmlNode) =
            let getT = getNodeText instr
            let getDouble x = 
                match getT x with
                | "" -> 0.0
                | value -> double <| value

            let expiry = getT "OptionExpiry"
            let level = getT "CalibrationLevel"
            let strike = getDouble "KRef"
            let vol = getDouble "BlackVol"
            let shift = getDouble "SmileShift"
            let smileSliceF = if (getT "SliceXType").Contains("Ratio") then (fun x -> x * (strike + shift) - shift) else (fun x -> x + strike)

            let strikes = getDelimNodeText instr ".//Buckets"
                                |> Array.map (double >> smileSliceF)                                
            let smiles =
                getNodes instr ".//OneDCurve" |> List.map (fun f -> 
                    (f.ParentNode.Name, getDelimNodeText f "Values" |> Array.map (double >> ((+) vol))))
                    |> List.filter (fun f -> (fst f).Contains("SmileSlice"))

            sbAppendSeq sb <| ["Level"; level]
            sbAppendSeq sb <| ["Expiry"; expiry]
            sbAppendSeq sb <| ["Kref"; string(strike)]
            sbAppendSeq sb <| ["SmileShift"; string(shift)]
            sbAppendSeq sb <| ["Vol"; string(vol)]
            printSabrParams instr
            sbAppendSeq sb <| "Strike" :: (smiles |> List.map fst)
            strikes |> Array.iteri (fun i e -> 
                sbAppendSeq sb <| e :: (smiles |> List.map (fun f -> (snd f).[i])))
            sb.AppendLine() |> ignore

        getNodes xmlFile "//CalibrationInstrument" |> List.iter printCalInstr
        sb.ToString()


    let getCurve (cNode : XmlNode) mainCurve = 
        let name = getNodeText cNode "CurveLabel"
        let buckets = getNodeText cNode "Buckets"
        let vals = getDelimNodeText cNode "Values" |> Array.map Double.Parse

        match mainCurve with
        | None       -> (name, buckets, vals)
        | Some(sprd) -> (name, buckets, vals|> Array.map2 (+) sprd)


    let parseCurves (xmlFile:XmlNode) =
        
        let mainCurve = getCurve <| getSingleNode xmlFile ".//RateCurve/OneDCurve" <| None
        let _, buckets, vals = mainCurve
        let sprdCurves = getNodes xmlFile ".//SpreadRateCurve/OneDCurve" 
                            |> List.map (fun f -> getCurve f (Some(vals)))
            
        let fst3 (a, _, _) = a
        let thrd (_, _, c) = c

        let data = mainCurve :: sprdCurves
        let rates = data |> List.map thrd

        let sb = new System.Text.StringBuilder()
        sbAppendSeq sb <| "Bucket" :: (data |> List.map fst3)
        buckets.Split(';') 
            |> Array.iteri 
                (fun i e -> sbAppendSeq sb <| e :: (rates |> List.map (fun a -> string <| a.[i])))
        sb.ToString()

    let parseCurvesOld (xmlFile:XmlNode) =

        let getCurve (cNode : XmlNode) mainCurve = 
            let tenor = getNodeText cNode "EvalSpreadCurveTenor"
            let name = cNode.Name + tenor
            let buckets = getNodeText cNode "./OneDCurve/Buckets"
            let vals = getDelimNodeText cNode "./OneDCurve/Values" |> Array.map Double.Parse

            match mainCurve with
            | None       -> (name, buckets, vals)
            | Some(sprd) -> (name, buckets, vals|> Array.map2 (+) sprd)
        
        let mainCurve = getCurve <| getSingleNode xmlFile "//RateCurve" <| None
        let _, buckets, vals = mainCurve
        let sprdCurves = 
            ["//DiscountSpreadCurve"; "//DiscountCmsSpreadCurve"] 
                |> Seq.map (getSingleNode xmlFile)
                |> Seq.append (getNodes xmlFile "//EvalSpreadCurve")
                |> Seq.filter (fun f -> f <> null)
                |> Seq.map (fun f -> getCurve f (Some(vals)))
                |> List.ofSeq
            
        let fst3 (a, _, _) = a
        let thrd (_, _, c) = c

        let data = mainCurve :: sprdCurves
        let rates = data |> List.map thrd

        let sb = new System.Text.StringBuilder()
        sbAppendSeq sb <| "Bucket" :: (data |> List.map fst3)
        buckets.Split(';') 
            |> Array.iteri 
                (fun i e -> sbAppendSeq sb <| e :: (rates |> List.map (fun a -> string <| a.[i])))
        sb.ToString()

    let parseCurvesFx (xmlFile:XmlNode) =

        let getCurve (cNode : XmlNode) mainCurve = 
            let name = getNodeText cNode "CurveLabel"
            let buckets = getNodeText cNode "Buckets"
            let vals = getDelimNodeText cNode "Values" |> Array.map Double.Parse

            match mainCurve with
            | None       -> (name, buckets, vals)
            | Some(sprd) -> (name, buckets, vals|> Array.map2 (+) sprd)
        
        let getCurvesByProcess (pNode:XmlNode) = 
            let mainCurve = getCurve <| getSingleNode pNode ".//RateCurve/OneDCurve" <| None
            let _, _, vals = mainCurve
            let sprdCurves = getNodes pNode ".//SpreadRateCurve/OneDCurve" 
                                |> List.map (fun f -> getCurve f (Some(vals)))
            mainCurve :: sprdCurves

        let printCurves (sb:System.Text.StringBuilder) (curves: (string * string * float[]) list) =
            let fst3 (a, _, _) = a
            let thrd (_, _, c) = c
            let _, buckets, _ = curves.Head
            sb.AppendLine("Bucket" + delimStr + buckets.Replace(';', delimChar)) |> ignore
            curves |> Seq.iter (fun f -> sbAppendSeq sb <| (fst3 f) :: (f |> thrd |> List.ofArray |> List.map string)) 
            
        let sb = new System.Text.StringBuilder()
        getNodes xmlFile "MacsML/ProcessesList/Processes/Process[contains(AssetType, 'Rate')]" 
            |> Seq.iter (getCurvesByProcess >> printCurves sb)
        sb.ToString()

    let parseSwapParams (xmlFile:XmlNode) =

        let parse2DCurve (n: XmlNode) = 
            let xaxis = getDelimNodeText n "TwoDCurve/Buckets"
            let yaxis = getDelimNodeText n "TwoDCurve/Values/OneDCurve/Buckets"
            let vals = getNodes n "TwoDCurve/Values/OneDCurve/Values" 
                            |> List.map xmlDelimNodeToArray
                            |> Array.ofList
            (xaxis, yaxis, vals)

        let swapParams = match getSingleNode xmlFile "//BGMTremorParams/SwapsParameters" with
                             | null -> []
                             | f    -> List.ofSeq <| Seq.cast<XmlNode>(f.ChildNodes)

        if Seq.isEmpty swapParams then
            String.Empty
        else
            let sb = new System.Text.StringBuilder()
            let data = swapParams |> List.map parse2DCurve
            sbAppendSeq sb <| "Expiry" :: "Tenor" :: (swapParams |> List.map (fun f -> f.Name))

            let maturs, tenors, _ = data.Head
            let thrd (_, _, c) = c
            let vals = data |> List.map thrd

            tenors |> Array.iteri (fun j tenor ->
                maturs |> Array.iteri (fun i matur ->
                    if (float(vals.Head.[i].[j]) <> 0.0 ) then
                        sbAppendSeq sb <| matur :: tenor :: (vals |> List.map (fun f -> f.[i].[j]))))

            sb.ToString()
    
    let parseLiborParams (xmlFile:XmlNode) =
        
        let parseMatrix (sb:System.Text.StringBuilder) (xmlNode:XmlNode) =
            let name = xmlNode.ParentNode.Name
            let libors = getDelimNodeText xmlNode "Buckets" |> Array.map (double >> int)
            let dates = (getNodeText xmlNode "./Values/OneDCurve/Buckets").Replace(';', '\t')
            let vals =  "./Values/OneDCurve/Values" 
                            |> getNodes xmlNode
                            |> List.map (fun f -> f.InnerText.Replace(';', '\t'))

            sbAppendLine sb <| name + delimStr + dates
            Seq.iter2 (fun x y -> sbAppendLine sb <| "L" + string(x) + delimStr + y) libors vals
            sbAppendLine sb ""
        
        let lpNode =  getSingleNode xmlFile "//BGMTremorParams/LiborsParameters"

        if (box lpNode) = null then
            String.Empty
        else
            let sb = new System.Text.StringBuilder()
            ".//LiborRate/OneDCurve" |> getSingleNode lpNode |> (fun n ->
                let dates = getDelimNodeText n "Buckets"
                let vals = getDelimNodeText n "Values"
                sbAppendLine sb <| "Date" + delimStr + "LiborRate"
                dates |> Seq.iter2 (fun a b -> sbAppendLine sb <| b + delimStr + a) vals
                sbAppendLine sb ""
                )

            ".//TwoDCurve" |> getNodes lpNode |> List.iter (parseMatrix sb)
            sb.ToString()  

    let parseCMSSO (xmlFile:XmlNode) =
        let nodes = getNodes xmlFile "//BGMCMSSOCalib//OneDCurve"
        if nodes.IsEmpty then
            String.Empty
        else
            let sb = new System.Text.StringBuilder()
            sbAppendSeq sb <| "Date" :: (nodes |> List.map (fun f -> f.ParentNode.Name))
            let dates = getDelimNodeText nodes.Head "Buckets"
            let vals = nodes |> List.map (getDelimNodeText >> (|>) "Values")
            dates 
                |> Array.iteri (fun i e -> sbAppendSeq sb <| e :: (vals |> List.map (Array.get >> (|>) i)))
            sb.ToString()

    let parseBGMCalibSmile (xmlFile:XmlNode) =
        let options = getNodes xmlFile "//BGMTremorParams/BGMSabrCalibSmile//Option"

        if options.IsEmpty then
            String.Empty
        else
            let sb = new System.Text.StringBuilder()
            let readOption (opt:XmlNode) = 
                ["Level"; "Maturity"; "Tenor"] |> List.iter (fun f -> sbAppendSeq sb <| f :: [getNodeText opt f])
                let smiles = ["SabrMarketSmile"; "SabrIntegratedParam"; "SabrTimeDepParam"] |> List.map (getSingleNode opt)
                sbAppendLine sb ""
                sbAppendSeq sb <| "Strike" :: (smiles |> List.map (fun f -> f.Name))
                let strikes = getDelimNodeText opt ".//Buckets"
                let vols = smiles |> List.map (getDelimNodeText >> (|>) ".//Values")
                strikes |> Array.iteri (fun i k -> sbAppendSeq sb <| k :: (vols |> List.map (fun f -> f.[i])))
                sbAppendLine sb ""
                ()
            options |> List.iter readOption
            sb.ToString()

    let parseMFNumeraire (xmlFile:XmlNode) =
        let mfNode = getSingleNode xmlFile "//MarkovFunctional"

        if (box mfNode = null) then
            String.Empty
        else
            let dates = getDelimNodeText mfNode "Dates"
            let mProcess = getDelimNodeText mfNode "NormalisedMarkovProcess"
            let numeraire = getNodes mfNode "NormalisedNumeraire" 
                               |> List.map xmlDelimNodeToArray

            let sb = new System.Text.StringBuilder()
            sbAppendSeq sb <| "Numerarire" :: List.ofArray mProcess
            numeraire |> List.iteri (fun i e -> sbAppendSeq sb <| dates.[i] :: List.ofArray e)
            sb.ToString()

    let parseMFDigitalSurface (xmlFile:XmlNode) =
        let data = 
            getNodes xmlFile "//CalibrationInstrument"
                |> List.map (fun f -> 
                                let T = getNodeText f "OptionExpiry"
                                let strikes = getDelimNodeText f "StrikeDigitMF"
                                let vols = getDelimNodeText f "ImpliedVolDigitMF"
                                (T, (strikes, vols))
                           )

        let expirires = data |> List.map fst
        let strikes = data |> List.map (snd >> fst)
        let vols = data |> List.map (snd >> snd)

        let sb = new System.Text.StringBuilder()
        let dummy = [0 .. strikes.Head.Length-1] |> List.map (fun f -> "K" + string(f))
        sbAppendSeq sb <| "Strikes" :: expirires
        dummy |> List.iteri (fun i e -> sbAppendSeq sb <| e :: (strikes |> List.map (Array.get >> ((|>) i))))
        sb.AppendLine() |> ignore
        
        sbAppendSeq sb <| "Volatilitites" :: expirires
        dummy |> List.iteri (fun i e -> sbAppendSeq sb <| e :: (vols |> List.map (Array.get >> ((|>) i))))
        sb.ToString()

    let parseMFSmiles (xmlFile:XmlNode) =
        
        let sb = new System.Text.StringBuilder()

        let printCalInstr (instr:XmlNode) =
            let getT = getNodeText instr
            
            let flds = ["Fictive"; "OptionExpiry"; "BlackVol"; "Forward"; "KRef"; "BasicDyn"; "CalibratedPremium"; "CalibratedVolatility"; "TargetVolatility"; "SmileShift"]
            let vals = flds |> List.map getT

            let vol = double <| vals.Item 2
            let fwd = double <| vals.Item 4

            let getV (pNode:XmlNode) (defV:float) (path:string) =
                match getSingleNode pNode path with
                    | null -> defV
                    | x -> double <| x.InnerText
                
            let kFactor, kShift, vShift = (getV instr 1.0 "SmileDynAdjusters/StrikeFactor", getV instr 0.0 "SmileDynAdjusters/StrikeShift", getV instr 0.0 "SmileDynAdjusters/VolShift")

            let strikes = getDelimNodeText instr ".//Buckets"
                                |> Array.map (fun f -> double(f)* fwd * kFactor + kShift)
            let smiles = 
                getNodes instr ".//OneDCurve" |> List.map (fun f -> 
                    (f.ParentNode.Name, getDelimNodeText f "Values" |> Array.map (double >> ((+) (vol+vShift)))))
                    |> List.filter (fun f -> (fst f).Contains("SmileSlice"))

            Seq.iter2 (fun a b -> sbAppendSeq sb <| [a; b]) flds vals

            sbAppendSeq sb <| "Strike" :: (smiles |> List.map fst)
            strikes |> Array.iteri (fun i e -> 
                sbAppendSeq sb <| e :: (smiles |> List.map (fun f -> (snd f).[i])))
            sb.AppendLine() |> ignore

        getNodes xmlFile "//CalibrationInstrument" |> List.iter printCalInstr
        sb.ToString()

    type GmpId = Discount | Forward | CrossBasket | CalBasket | CalBGM | CalSwapBGM | CalSabrBGM | CalCmsBGM | Vol_CalBasket | Sabr_VVol_CalBsk | SmileDeformation with
        member this.ToString = Toolbox.DUnions.toStr this
        static member fromStr s = Toolbox.DUnions.fromStr<GmpId> s
        static member listMembers = Toolbox.DUnions.listMembers<GmpId> 

    let parseGmpField (xmlFile:XmlNode) (fldType:GmpId) (level:int) =

        let parseGmpField (gmpNode:XmlNode) =
            let getNamed n (a, b) = (getNodeText n a,  getDelimNodeText n b)
            let getBreakdowns (str:string) = 
                let (lab, vals) = getNamed gmpNode <| ("Label" + str, "Vector" + str)
                (lab, vals, vals.Length)

            let (xLab, xVals, xLen) = getBreakdowns "X"
            let (yLab, yVals, yLen) = getBreakdowns "Y"
            let (zLab, zVals, zLen) = getBreakdowns "Z"

            let data = getNodes gmpNode "FieldsLists/Fields/Field" 
                       |> List.map (getNamed >> ((|>) ("FieldName", "Values")))
                       |> List.unzip

            let sb = new System.Text.StringBuilder()

            if (zLen = 0) then
                if (yLen = 0) then
                    sbAppendSeq sb <| xLab :: fst data
                    xVals |> Array.iteri (fun i x -> sbAppendSeq sb <| x :: (snd data |> List.map (Array.get >> ((|>) i))))
                else
                    sbAppendSeq sb <| xLab :: yLab :: fst data
                    xVals |> Array.iteri (fun i x -> 
                        yVals |> Array.iteri (fun j y ->
                                sbAppendSeq sb <| x :: y :: 
                                    (snd data |> List.map (Array.get >> ((|>) (i * yLen  + j ))))))
            else
                sbAppendSeq sb <| xLab :: yLab :: zLab :: fst data
                xVals |> Array.iteri (fun i x -> 
                    yVals |> Array.iteri (fun j y ->
                        zVals |> Array.iteri (fun k z ->
                            sbAppendSeq sb <| x :: y :: z ::
                                (snd data |> List.map (Array.get >> ((|>) ((i * yLen  + j ) * zLen + k)))))))

            sb.ToString()
             

        let gmpNodes = getNodes xmlFile (sprintf "//GmpField[contains(GmpID,'%A')]" fldType) |> Array.ofList
        match gmpNodes with
        | x when level < 0 || level > x.Length - 1 -> String.Empty
        | _ -> parseGmpField gmpNodes.[level]

    let parseQueryRes (xmlFile:XmlNode) =
        
        let getVarDescr (qNode:XmlNode) =
            qNode 
                |> (getNodes >> (|>) ".//VariableDescription") 
                |> Seq.map (fun n -> n.InnerText.Split(';').[0])
                |> List.ofSeq

        let qNodes = getNodes xmlFile "MacsML/QueriesList/Queries/Query"

        let sb = new System.Text.StringBuilder()
        sbAppendSeq sb <| "QueryType" :: "Curve" :: "Bucket" :: "Shock" :: (getVarDescr <| first qNodes)
            
        qNodes 
            |> Seq.iter (fun qNode -> 
                let qname = getNodeText qNode "QRType"
                let curveLbl = getNodeText qNode "QRSpreadCurveLabel"
                let buckets = getDelimNodeText qNode "QRBuckets/OneDCurve/Buckets"
                let values = getDelimNodeText qNode "QRBuckets/OneDCurve/Values"
                let qRes = getNodes qNode "QRResultsList/QRResults/QRResult"
                let bucketed = (int(getNodeText qNode "TermStructureAsOneBump") <> 1)
                let isBump = buckets.Length > 0

                for i = 0 to qRes.Length - 1 do
                    let mutable nodes = getNodes qRes.[i] ".//VariableValue"
                    if nodes.Length = 0 then
                        nodes <- getNodes qRes.[i] ".//Value"
                    let vals = nodes |> List.map (fun f -> f.InnerText.Trim())
                    let date = if isBump && bucketed then buckets.[i] else ""
                    let shock = if isBump then values.[i] else ""
                    sbAppendSeq sb <| qname :: curveLbl :: date :: shock :: vals
                    )
        sb.ToString()       


    let parseQueryResBulk (xmlFile:XmlNode) =
        
        let qNodes = getNodes xmlFile "MacsML/QueriesList/Queries/Query"

        let sb = new System.Text.StringBuilder()
        sbAppendSeq sb <| "QueryType" :: "Bucket" :: ["Shock"]
            
        qNodes 
            |> Seq.iter (fun qNode -> 
                let qname = getNodeText qNode "QRType"
                let buckets = getDelimNodeText qNode "QRBuckets/OneDCurve/Buckets"
                let values = getDelimNodeText qNode "QRBuckets/OneDCurve/Values"
                let qRes = getNodes qNode "QRResultsList/QRResults//Option"
                let bucketed = (int(getNodeText qNode "TermStructureAsOneBump") <> 1)
                let isBump = buckets.Length > 0

                for i = 0 to qRes.Length - 1 do
                    let vals = getNodes qRes.[i] ".//Value" |> List.map (fun f -> f.InnerText)
                    let date = if isBump && bucketed then buckets.[i] else ""
                    let shock = if isBump then values.[i] else ""
                    sbAppendSeq sb <| qname :: date :: shock :: vals
                    )
        sb.ToString()   


    let dqParseSwapParams (xmlFile:XmlNode) =
        //  Read 2D params
        let parse2DCurve (n: XmlNode) = 
            let xaxis = getDelimNodeText n "TwoDCurve/Buckets"
            let yaxis = getDelimNodeText n "TwoDCurve/Values/OneDCurve/Buckets"
            let vals = getNodes n "TwoDCurve/Values/OneDCurve/Values" 
                            |> List.map xmlDelimNodeToArray
                            |> Array.ofList
            (xaxis, yaxis, vals)

        //  
        let getSwapParamsNodes (n: XmlNode) = 
            match getSingleNode n ".//BGMTremorParams/SwapsParameters" with
                | null -> []
                | f    -> List.ofSeq <| Seq.cast<XmlNode>(f.ChildNodes)

        let getSPNames (qNode:XmlNode) =
            qNode 
                |> (getSingleNode >> (|>) "BucketsProcessesList//Process") 
                |> getSwapParamsNodes
                |> Seq.map (fun n -> n.Name)
                |> List.ofSeq

        let qNodes = getNodes xmlFile "QueriesList/Queries/Query"

        let sb = new System.Text.StringBuilder()
        sbAppendSeq sb <| "QueryType" :: "Bucket" :: "Shock" :: "Expiry" :: "Tenor" :: (getSPNames <| first qNodes)
            
        qNodes 
            |> Seq.iter (fun qNode -> 
                            let qname = getNodeText qNode "QRType"
                            let buckets = getDelimNodeText qNode "QRBuckets/OneDCurve/Buckets"
                            let values = getDelimNodeText qNode "QRBuckets/OneDCurve/Values"
                            let qProcesses = getNodes qNode "BucketsProcessesList//Process"
                            let bucketed = (int(getNodeText qNode "TermStructureAsOneBump") <> 1)
                            let isBump = buckets.Length > 0

                            for i = 0 to qProcesses.Length - 1 do

                                let date = if isBump && bucketed then buckets.[i] else "0.0"
                                let shock = if isBump then (if bucketed then values.[i] else values.[values.Length-1]) else "0.0"

                                let swapParams = getSwapParamsNodes qProcesses.[i]
                    
                                if Seq.isEmpty swapParams then
                                    ()
                                else
                                    let data = swapParams |> List.map parse2DCurve

                                    let maturs, tenors, _ = data.Head
                                    let thrd (_, _, c) = c
                                    let vals = data |> List.map thrd

                                    tenors |> Array.iteri (fun j tenor ->
                                        maturs |> Array.iteri (fun i matur ->                                            
                                            if (float(vals.Head.[i].[j]) <> 0.0 ) then
                                                sbAppendSeq sb <| qname :: date :: shock :: matur :: tenor :: (vals |> List.map (fun f -> f.[i].[j]))))
                        )
        sb.ToString()

    let getBGMCalibSmile (xmlFile:XmlNode) =
        let options = getNodes xmlFile "//BGMTremorParams/BGMSabrCalibSmile//Option"

        if options.IsEmpty then
            raise (new Exception("No data!"))
        else            
            let readOption (opt:XmlNode) = 
                let level = getNodeText opt "Level" |> int
                let maturity = getNodeText opt "Maturity" |> float
                let tenor = getNodeText opt "Tenor" |> float
                let smiles = ["SabrMarketSmile"; "SabrIntegratedParam"; "SabrTimeDepParam"] |> List.map (getSingleNode opt)

                let strikes = getDelimNodeText opt ".//Buckets" |> Array.map float
                let vols = smiles |> List.map (fun f -> getDelimNodeText f ".//Values" |> Array.map float)
                (level, maturity, tenor, strikes, vols)

            options |> List.map readOption

    let private pParseBGMCalibSmileErr (sb:System.Text.StringBuilder) (printHeader:bool) (headerPrefix:string list) (prefix:string list) (xmlFile:XmlNode) =

        let options = getNodes xmlFile ".//BGMTremorParams/BGMSabrCalibSmile//Option"

        if options.IsEmpty then
            ()
        else
            let h1 = ["Level"; "Maturity"; "Tenor"]
            
            let readOption (opt:XmlNode) = 
                
                let getErr (xArr:float[]) (yArr:float[]) =
                    let err = Array.map2 (fun x y -> abs(x - y)*1e4) xArr yArr
                    let arr3p = Array.sub err 1 3
                    let errAtm = err.[2]
                    let err3p = sqrt((Array.map2 (*) arr3p arr3p) |> Array.average)
                    let err5p = sqrt((Array.map2 (*) err err) |> Array.average)
                    [errAtm; err3p; err5p] |> List.map string


                let h1Val = h1 |> List.map (getNodeText opt)                              
                let smiles = ["SabrMarketSmile"; "SabrIntegratedParam"; "SabrTimeDepParam"] 
                                |> List.map (fun f -> getDelimNodeText opt (f + "/OneDCurve/Values") |> Array.map float)
                let errMkt = getErr smiles.[0] smiles.[1]
                let errCal = getErr smiles.[1] smiles.[2]

                sbAppendSeq sb <|  prefix @ h1Val @ errMkt @ errCal
                
            if printHeader then
                sbAppendSeq sb <| headerPrefix @ h1 @ ["MkrErrAtm"; "MkrErr3p"; "MkrErr5p"; "CalErrAtm"; "CalErr3p"; "CalErr5p"]  

            options |> List.iter readOption

    let parseBGMCalibSmileErr (xmlFile:XmlNode) =
        let sb = new System.Text.StringBuilder()
        pParseBGMCalibSmileErr sb true [] [] xmlFile
        sb.ToString()          

    let dqParseBGMSmilesErr (xmlFile:XmlNode) =

        let qNodes = getNodes xmlFile "QueriesList/Queries/Query"

        let sb = new System.Text.StringBuilder()
        let headerPrefix = ["QueryType"; "Bucket"; "Shock"]
        let isFirst = ref true

        qNodes 
            |> Seq.iter (fun qNode -> 
                            let qname = getNodeText qNode "QRType"
                            let buckets = getDelimNodeText qNode "QRBuckets/OneDCurve/Buckets"
                            let values = getDelimNodeText qNode "QRBuckets/OneDCurve/Values"
                            let qProcesses = getNodes qNode "BucketsProcessesList//Process"
                            let bucketed = (int(getNodeText qNode "TermStructureAsOneBump") <> 1)
                            let isBump = buckets.Length > 0

                            for i = 0 to qProcesses.Length - 1 do

                                let date = if isBump && bucketed then buckets.[i] else "0.0"
                                let shock = if isBump then (if bucketed then values.[i] else values.[values.Length-1]) else "0.0"
                                if !isFirst then
                                    isFirst:= false
                                    pParseBGMCalibSmileErr sb true headerPrefix [qname; date; shock] qProcesses.[i]
                                else
                                    pParseBGMCalibSmileErr sb false [] [qname; date; shock] qProcesses.[i]
                        )
        sb.ToString()


    let macsiParseSchedule (xmlFile:XmlNode) = 
        
        let infoFlds = ["TradeNumber"; "HorizonDate" ]
        let legFlds = ["StartDates"; "EndDates"; "FixingDates"; "PaymentDates" ]
        let exerciseFlds = ["ExerciseDate"; "EffectiveDate"]


        let infoData = infoFlds |> List.map (fun f ->  getNodeText xmlFile ("MacsIML/" + f))

        let legData = "MacsIML/FinancialDefinition/PayoffDetails/LegDetails/Legs/Leg" 
                        |> getNodes xmlFile
                        |> List.map (fun n -> legFlds |> List.map (fun k -> (getDelimNodeText n k).[0]))
                        |> List.concat 

        let exerciseData = exerciseFlds |> List.map (getNodeText (getSingleNode xmlFile "MacsIML/FinancialDefinition/PayoffDetails/ExerciseDetails/ExerciseClauses/ExerciseClause"))
                             
        let sb = new System.Text.StringBuilder()
        sbAppendSeq sb <| List.concat [infoFlds; legFlds; legFlds; exerciseFlds]
        sbAppendSeq sb <| List.concat [infoData; legData; exerciseData]
        sb.ToString()

    let private parseFlowsHelper (xmlFile:XmlNode) (fdate:string) (pdate:string) = 
        let tmp = ["IDLeg"; "CashFlowTypology"; "Expectation"; "ConditionalDensity"; "ExpectationExerciseDate"]
        let header = "FixingDate" :: "PayDate" :: tmp
        let fields = fdate :: pdate :: tmp 
        let sb = new System.Text.StringBuilder()
        sbAppendSeq sb <| "Variable" :: header
        @"MacsML/QueriesList/Queries/Query/QRResultsList/QRResults//Variable[MCDensity]"
            |> getNodes xmlFile
            |> Seq.iter (fun f -> 
                                let variable = ("VariableDescription" |> getNodeText f).Split(';').[0].Trim()
                                "./MCDensity/Density" |> getNodes f |> Seq.iter (fun f -> sbAppendSeq sb <| variable :: List.map (getNodeText f) fields))
        sb.ToString()

    let parseFlows (xmlFile:XmlNode) = 
        parseFlowsHelper xmlFile "FixingDate"  "PayDate"

    let parseFlowsOld (xmlFile:XmlNode) = 
        parseFlowsHelper xmlFile "Dates" "FixingDate"

    let parseTsrFlows (xmlFile:XmlNode) = 
        let header = ["ID"; "A"; "B"; "C"; "Flow"; "FixingDate"; "PmntDate"]
        let sb = new System.Text.StringBuilder()
        sbAppendSeq sb <| header
        @"MacsML/ExptectedCashFlows/ExptectedCashFlowsList/Flow"
            |> getNodes xmlFile
            |> Seq.map xmlDelimNodeToArray
            |> Seq.iter (sbAppendSeq sb)
        sb.ToString()

    
    let parseTsrVega (xmlFile:XmlNode) = 
        let header = ["A"; "B"; "C"; "FixingDate"; "Tenor"; "D"; "Strike"; "Vega"]
        let sb = new System.Text.StringBuilder()
        sbAppendSeq sb <| header
        @"MacsML/QueriesList/Queries/Query/QRResultsList/QRResults//VariableSns/VariableSnsList/ASns"
            |> getNodes xmlFile
            |> Seq.map xmlDelimNodeToArray
            |> Seq.iter (sbAppendSeq sb)
        sb.ToString()

    let parsePastFlows (xmlFile:XmlNode) = 
        let nodes = getNodes xmlFile @"MacsML/PastCashFlowsInfo/FlowsList/Flows/Flow"
        if nodes.Length = 0 then
            String.Empty
        else
            let sb = new System.Text.StringBuilder()
            let header = nodes.[0].ChildNodes |> Seq.cast<XmlNode> |> Seq.map (fun f -> f.Name)
            sbAppendSeq sb <| header
            nodes
                |> Seq.iter (fun f -> f.ChildNodes |> Seq.cast<XmlNode> |> Seq.map (fun f -> f.InnerText) |> sbAppendSeq sb)
            sb.ToString()
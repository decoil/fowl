/// Physics Simulation Example
/// Projectile motion and pendulum dynamics

module PhysicsSimulationExample

open System
open Fowl
open Fowl.Core
open Fowl.Stats

/// Projectile state: position and velocity
type ProjectileState = {
    Time: float
    X: float
    Y: float
    Vx: float
    Vy: float
}

/// Pendulum state
type PendulumState = {
    Time: float
    Theta: float    // Angle in radians
    Omega: float    // Angular velocity
}

/// Simulate projectile motion with air resistance
let simulateProjectile (v0: float) (angle: float) (h0: float)
                       (g: float) (drag: float) (dt: float) 
                       : ProjectileState[] =
    // Initial conditions
    let mutable state = {
        Time = 0.0
        X = 0.0
        Y = h0
        Vx = v0 * cos angle
        Vy = v0 * sin angle
    }
    
    let trajectory = ResizeArray<ProjectileState>()
    trajectory.Add(state)
    
    // Euler integration
    while state.Y >= 0.0 do
        let v = sqrt (state.Vx ** 2.0 + state.Vy ** 2.0)
        
        // Accelerations
        let ax = -drag * v * state.Vx
        let ay = -g - drag * v * state.Vy
        
        // Update
        state <- {
            Time = state.Time + dt
            X = state.X + state.Vx * dt
            Y = state.Y + state.Vy * dt
            Vx = state.Vx + ax * dt
            Vy = state.Vy + ay * dt
        }
        
        trajectory.Add(state)
    
    trajectory.ToArray()

/// Simulate simple pendulum
let simulatePendulum (theta0: float) (omega0: float) (length: float)
                     (g: float) (dt: float) (duration: float)
                     : PendulumState[] =
    let nSteps = int (duration / dt) + 1
    let trajectory = Array.zeroCreate nSteps
    
    let mutable state = {
        Time = 0.0
        Theta = theta0
        Omega = omega0
    }
    
    trajectory.[0] <- state
    
    // Semi-implicit Euler (more stable)
    for i = 1 to nSteps - 1 do
        let alpha = -g / length * sin state.Theta
        
        state <- {
            Time = float i * dt
            Omega = state.Omega + alpha * dt
            Theta = state.Theta + state.Omega * dt
        }
        
        trajectory.[i] <- state
    
    trajectory

/// Calculate period from trajectory
let calculatePeriod (trajectory: PendulumState[]) : float =
    // Find zero crossings (theta changing sign)
    let crossings = 
        trajectory
        |> Array.pairwise
        |> Array.indexed
        |> Array.filter (fun (_, (s1, s2)) -
            s1.Theta * s2.Theta < 0.0 && s1.Omega > 0.0)
        |> Array.map (fun (i, _) -> trajectory.[i].Time)
    
    if crossings.Length >= 2 then
        // Average period from consecutive crossings
        let periods = 
            crossings
            |> Seq.pairwise
            |> Seq.map (fun (t1, t2) -> 2.0 * (t2 - t1))
            |> Seq.toArray
        Array.average periods
    else
        0.0

/// Calculate projectile range
let calculateRange (trajectory: ProjectileState[]) : float =
    // Find last point before hitting ground
    let lastAirborne = 
        trajectory
        |> Array.filter (fun s -> s.Y >= 0.0)
        |> Array.tryLast
    
    match lastAirborne with
    | Some s -> s.X
    | None -> 0.0

/// Calculate max height
let calculateMaxHeight (trajectory: ProjectileState[]) : float =
    trajectory
    |> Array.maxBy (fun s -> s.Y)
    |> fun s -> s.Y

/// Run physics simulations
let runPhysics() : unit =
    printfn "=== Physics Simulation Examples ==="
    printfn ""
    
    // Example 1: Projectile Motion
    printfn "1. Projectile Motion"
    printfn "   Initial velocity: 25 m/s"
    printfn "   Launch angle: 45 degrees"
    printfn "   Initial height: 2 meters"
    printfn "   Drag coefficient: 0.01"
    printfn ""
    
    let v0 = 25.0
    let angle = Math.PI / 4.0
    let h0 = 2.0
    let g = 9.81
    let drag = 0.01
    let dt = 0.01
    
    let projTraj = simulateProjectile v0 angle h0 g drag dt
    let range = calculateRange projTraj
    let maxHeight = calculateMaxHeight projTraj
    let flightTime = (Array.last projTraj).Time
    
    printfn "   Results:"
    printfn "     Range: %.2f meters" range
    printfn "     Max height: %.2f meters" maxHeight
    printfn "     Flight time: %.2f seconds" flightTime
    printfn ""
    
    // Without drag comparison
    let projNoDrag = simulateProjectile v0 angle h0 g 0.0 dt
    let rangeNoDrag = calculateRange projNoDrag
    
    printfn "   Comparison (no drag):"
    printfn "     Range: %.2f meters" rangeNoDrag
    printfn "     Drag reduces range by %.1f%%" 
            ((1.0 - range/rangeNoDrag) * 100.0)
    printfn ""
    
    // Example 2: Pendulum
    printfn "2. Simple Pendulum"
    printfn "   Length: 1 meter"
    printfn "   Initial angle: 45 degrees"
    printfn "   Initial angular velocity: 0"
    printfn ""
    
    let theta0 = Math.PI / 4.0
    let omega0 = 0.0
    let length = 1.0
    let duration = 10.0
    
    let pendTraj = simulatePendulum theta0 omega0 length g dt duration
    let period = calculatePeriod pendTraj
    let theoreticalPeriod = 2.0 * Math.PI * sqrt (length / g)
    
    // Small angle approximation
    let smallAnglePeriod = 2.0 * Math.PI * sqrt (length / g)
    
    printfn "   Results:"
    printfn "     Measured period: %.4f seconds" period
    printfn "     Theoretical (small angle): %.4f seconds" theoreticalPeriod
    printfn "     Error: %.2f%%" (abs (period - theoreticalPeriod) / theoreticalPeriod * 100.0)
    printfn ""
    
    // Example 3: Energy analysis
    printfn "3. Energy Conservation Check"
    printfn ""
    
    let mass = 1.0
    let energies = 
        pendTraj
        |> Array.map (fun s -
            let v = s.Omega * length
            let h = length * (1.0 - cos s.Theta)
            let ke = 0.5 * mass * v * v
            let pe = mass * g * h
            ke + pe)
    
    let! eMean = Descriptive.mean energies
    let! eStd = Descriptive.std energies
    
    printfn "   Total mechanical energy:"
    printfn "     Mean: %.6f J" eMean
    printfn "     Std: %.6f J" eStd
    printfn "     Variation: %.4f%%" (eStd / eMean * 100.0)
    
    if eStd / eMean < 0.01 then
        printfn "     ✅ Energy conserved (within numerical error)"
    else
        printfn "     ⚠️ Energy not well conserved"
    printfn ""
    
    // Example 4: Different angles
    printfn "4. Period vs Initial Angle"
    printfn ""
    printfn "   Angle (deg) │ Period (s) │ Deviation from small-angle"
    printfn "   ────────────┼────────────┼───────────────────────────"
    
    let angles = [|5.0; 15.0; 30.0; 45.0; 60.0; 75.0; 90.0|]
    
    angles |> Array.iter (fun deg -
        let theta = deg * Math.PI / 180.0
        let traj = simulatePendulum theta 0.0 length g dt 20.0
        let p = calculatePeriod traj
        let deviation = (p - smallAnglePeriod) / smallAnglePeriod * 100.0
        printfn "   %6.1f      │ %10.4f │ %10.2f%%" deg p deviation)
    
    printfn ""
    printfn "   Note: Period increases with amplitude (nonlinear effect)"
    printfn ""
    
    printfn "=== Physics Simulation Complete ==="

[<EntryPoint>]
let main argv =
    runPhysics()
    0
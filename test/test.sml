structure Tests = struct open Harness structure P = Particle

fun run () = let
  val () = section "particle vs Kalman (true state = 1.0)"
  val (k, p) = P.compareKalman ()
  val () = checkRealTol 0.1 "kalman  estimate near 1" (1.0, k)
  val () = checkRealTol 0.3 "particle estimate near 1" (1.0, p)

  val () = section "filterStep single observation"
  val fs = P.filterStep 1.0 0.0
  val () = check "filterStep shifts toward obs" (fs > 0.3)

  (* ---- weighted statistics ---- *)
  val () = section "weighted statistics"
  val vals = [1.0, 2.0, 3.0, 4.0]
  val wUnif = [0.25, 0.25, 0.25, 0.25]
  val () = checkRealTol 1E~9 "uniform weighted mean = 2.5" (2.5, P.weightedMean vals wUnif)
  (* variance of {1,2,3,4} (population) = 1.25 *)
  val () = checkRealTol 1E~9 "uniform weighted var = 1.25" (1.25, P.weightedVariance vals wUnif)
  (* skewed weights toward 4.0 *)
  val wSkew = [0.1, 0.1, 0.1, 0.7]
  val () = check "skewed mean > 3" (P.weightedMean vals wSkew > 3.0)

  val () = section "effective sample size"
  (* uniform weights over N => ESS = N *)
  val () = checkRealTol 1E~9 "ESS uniform = 4" (4.0, P.effectiveSampleSize wUnif)
  (* a single dominating weight => ESS ~ 1 *)
  val () = checkRealTol 1E~6 "ESS degenerate ~ 1" (1.0, P.effectiveSampleSize [1.0,0.0,0.0,0.0])

  (* ---- determinism: same seed => identical result ---- *)
  val () = section "fixed-seed determinism"
  val obs = [1.0, 0.9, 1.1, 1.0]
  val a1 = P.runSIR obs 0.0 0wx1234
  val a2 = P.runSIR obs 0.0 0wx1234
  val () = check "same seed => identical" (Real.== (a1, a2))
  val b1 = P.runSIR obs 0.0 0wx9999
  val () = check "different seed => (usually) different" (not (Real.== (a1, b1)))

  (* ---- trace records: weights normalized, ESS within bounds ---- *)
  val () = section "trace invariants"
  val params = { n = 100, q = 0.01, r = 0.1, initVar = 1.0 }
  val trace = P.runSIRTrace params obs 0.0 0wx2024
  val () = checkInt "one step per obs" (4, List.length trace)
  val () = List.app (fn (st : P.step) =>
    let val sumW = List.foldl op+ 0.0 (#weights st)
    in check "weights sum to 1" (Real.abs (sumW - 1.0) <= 1E~6) end) trace
  val () = List.app (fn (st : P.step) =>
    check "ESS in (0, N]" (#ess st > 0.0 andalso #ess st <= 100.0 + 1E~6)) trace
  val () = List.app (fn (st : P.step) =>
    checkInt "particle count = N" (100, List.length (#particles st))) trace

  (* ---- convergence: final mean tracks the true state ---- *)
  val () = section "convergence"
  val longObs = List.tabulate (30, fn _ => 2.0)
  val est = P.runSIRWith params longObs 0.0 0wx55
  val () = checkRealTol 0.4 "converges to 2.0" (2.0, est)

  (* ---- configurable N: bigger N reduces estimator spread ---- *)
  val () = section "configurable N"
  val smallP = { n = 20,  q = 0.01, r = 0.1, initVar = 1.0 }
  val bigP   = { n = 500, q = 0.01, r = 0.1, initVar = 1.0 }
  val eSmall = P.runSIRWith smallP longObs 0.0 0wx7
  val eBig   = P.runSIRWith bigP   longObs 0.0 0wx7
  val () = check "small N produces a finite estimate" (Real.isFinite eSmall)
  val () = checkRealTol 0.5 "big N near truth" (2.0, eBig)

in Harness.run () end end

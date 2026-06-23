structure Tests = struct open Harness structure P = Particle
fun run () = let
  val () = section "particle vs Kalman (true state = 1.0)"
  val (k, p) = P.compareKalman ()
  (* Both filters run on obs=[1.0]*5 from x0=0, process/obs noise Q=0.01,R=0.1.
     The Kalman converges quickly; the SIR particle filter should also track
     the true state within 0.3. *)
  val () = checkRealTol 0.1 "kalman  estimate near 1" (1.0, k)
  val () = checkRealTol 0.3 "particle estimate near 1" (1.0, p)

  (* filterStep: single SIR step observing z=1.0 from x=0.0.
     Posterior mean should shift toward 1.0. *)
  val () = section "filterStep single observation"
  val fs = P.filterStep 1.0 0.0
  val () = check "filterStep shifts toward obs" (fs > 0.3)
in Harness.run () end end

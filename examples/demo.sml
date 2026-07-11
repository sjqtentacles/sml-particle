(* demo.sml - drives the SIR particle filter and the 1-D Kalman filter over a
   fixed observation sequence, exercises the weighted-statistics helpers, and
   prints a short SIR convergence trace. All randomness comes from literal
   fixed seeds; reals are formatted with a fixed digit count (and negative
   zero normalized) so MLton and Poly/ML agree byte-for-byte. *)

structure P = Particle

fun fmtR x =
    let val x = if Real.== (x, 0.0) then 0.0 else x
    in Real.fmt (StringCvt.FIX (SOME 4)) x end

val () = print "sml-particle demo\n"
val () = print "==================\n\n"

(* -- particle filter vs Kalman filter, five observations of a state near 1.0 -- *)
val obs1 = [1.0, 1.0, 1.0, 1.0, 1.0]
val kalmanEst = P.runKalman obs1 0.0
val particleEst = P.runSIR obs1 0.0 0wx1234
val () = print ("kalman estimate       = " ^ fmtR kalmanEst ^ "\n")
val () = print ("particle estimate     = " ^ fmtR particleEst ^ "\n\n")

(* -- weighted statistics over a literal value/weight batch -- *)
val vals = [1.0, 2.0, 3.0, 4.0]
val wts  = [0.1, 0.2, 0.3, 0.4]
val () = print ("weighted mean         = " ^ fmtR (P.weightedMean vals wts) ^ "\n")
val () = print ("weighted variance     = " ^ fmtR (P.weightedVariance vals wts) ^ "\n")
val () = print ("effective sample size = " ^ fmtR (P.effectiveSampleSize wts) ^ "\n\n")

(* -- convergence: a small particle cloud tracking a step from 0.0 to 2.0 -- *)
val smallParams = { n = 50, q = 0.01, r = 0.1, initVar = 1.0 }
val obs2 = List.tabulate (6, fn _ => 2.0)
val trace = P.runSIRTrace smallParams obs2 0.0 0wx55
val () = print "SIR trace (n=50, converging toward 2.0):\n"
val () =
    List.app
      (fn (i, st : P.step) =>
          print ("  step " ^ Int.toString i ^ ": mean = " ^ fmtR (#mean st)
                 ^ ", ess = " ^ fmtR (#ess st) ^ "\n"))
      (List.tabulate (List.length trace, fn i => (i, List.nth (trace, i))))

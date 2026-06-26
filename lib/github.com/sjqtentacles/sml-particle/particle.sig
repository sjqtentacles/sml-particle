signature PARTICLE =
sig
  (* Model / filter parameters for the 1-D random-walk tracker:
       x_{t+1} = x_t + w,  w ~ N(0, q)     (process)
       z_t     = x_t + v,  v ~ N(0, r)     (measurement)
     n         = particle count
     initVar   = variance of the initial particle cloud around x0 *)
  type params = { n : int, q : real, r : real, initVar : real }

  (* q=0.01, r=0.1, n=200, initVar=1.0 *)
  val defaultParams : params

  (* The result of one SIR step (post-resample, weights are uniform after
     resampling so we report the PRE-resample weights and summaries). *)
  type step =
    { particles : real list   (* resampled particles *)
    , weights   : real list   (* normalized pre-resample weights *)
    , mean      : real         (* weighted mean estimate *)
    , ess       : real }       (* effective sample size *)

  (* ---- weighted statistics ---- *)
  val weightedMean     : real list -> real list -> real   (* values weights *)
  val weightedVariance : real list -> real list -> real
  val effectiveSampleSize : real list -> real             (* normalized weights *)

  (* ---- filters ---- *)

  (* Run an SIR filter with explicit params; returns the final weighted mean. *)
  val runSIRWith : params -> real list -> real -> Word64.word -> real
  (* Run with defaultParams. *)
  val runSIR     : real list -> real -> Word64.word -> real
  (* Run and return the per-observation step record trace. *)
  val runSIRTrace: params -> real list -> real -> Word64.word -> step list
  (* 1-D Kalman filter, final estimate. *)
  val runKalman  : real list -> real -> real

  (* ---- legacy convenience ---- *)
  val filterStep    : real -> real -> real
  val compareKalman : unit -> real * real
end

(* SIR particle filter + Kalman comparison for 1-D random-walk tracking.

   Model:   x_{t+1} = x_t + w,   w ~ N(0, q)   (process)
            z_t     = x_t + v,   v ~ N(0, r)   (measurement)

   Particles are stored in an array and updated in place; the PRNG state is
   threaded purely so runs are byte-reproducible from a seed. *)

structure Particle :> PARTICLE =
struct
  type params = { n : int, q : real, r : real, initVar : real }

  val defaultParams = { n = 200, q = 0.01, r = 0.1, initVar = 1.0 }

  type step =
    { particles : real list, weights : real list, mean : real, ess : real }

  (* Box-Muller: one standard Gaussian from two U[0,1) samples *)
  fun gauss1 rng =
    let
      val (u1, rng1) = Prng.SplitMix64.real01 rng
      val (u2, rng2) = Prng.SplitMix64.real01 rng1
      val mag = Math.sqrt (~2.0 * Math.ln (Real.max (u1, 1E~300)))
    in (mag * Math.cos (2.0 * Math.pi * u2), rng2) end

  (* ---- weighted statistics ---- *)

  fun weightedMean values weights =
    let
      val tot = List.foldl op+ 0.0 weights
      val s = ListPair.foldl (fn (v, w, acc) => acc + v * w) 0.0 (values, weights)
    in if Real.== (tot, 0.0) then 0.0 else s / tot end

  fun weightedVariance values weights =
    let
      val m = weightedMean values weights
      val tot = List.foldl op+ 0.0 weights
      val s = ListPair.foldl (fn (v, w, acc) => acc + w * (v - m) * (v - m)) 0.0 (values, weights)
    in if Real.== (tot, 0.0) then 0.0 else s / tot end

  (* ESS = 1 / sum(w_i^2) for normalized weights. *)
  fun effectiveSampleSize weights =
    let val s2 = List.foldl (fn (w, acc) => acc + w * w) 0.0 weights
    in if Real.== (s2, 0.0) then 0.0 else 1.0 / s2 end

  (* ---- one SIR step over an array; returns (step-record, next rng) ---- *)
  fun sirOneStep (q, r) parts z rng =
    let
      val n = Array.length parts

      (* Predict: add process noise *)
      val rng =
        Array.foldli (fn (i, x, rng) =>
          let val (g, rng') = gauss1 rng
          in Array.update (parts, i, x + g * Math.sqrt q); rng' end)
          rng parts

      (* Log-weights then normalize *)
      val logW = Array.tabulate (n, fn i =>
        ~0.5 * (z - Array.sub (parts, i)) * (z - Array.sub (parts, i)) / r)
      val maxLW = Array.foldl Real.max (Array.sub (logW, 0)) logW
      val w = Array.tabulate (n, fn i => Math.exp (Array.sub (logW, i) - maxLW))
      val tot = Array.foldl op+ 0.0 w
      val () = Array.modify (fn wi => wi / tot) w

      (* summaries computed from PRE-resample particles + weights *)
      val partsList = List.tabulate (n, fn i => Array.sub (parts, i))
      val wList = List.tabulate (n, fn i => Array.sub (w, i))
      val mean = weightedMean partsList wList
      val ess = effectiveSampleSize wList

      (* Cumulative weights for systematic resampling *)
      val cumW = Array.array (n, 0.0)
      val () = Array.update (cumW, 0, Array.sub (w, 0))
      val () = List.app (fn i =>
        Array.update (cumW, i, Array.sub (cumW, i - 1) + Array.sub (w, i)))
        (List.tabulate (n - 1, fn k => k + 1))

      val (u0, rng') = Prng.SplitMix64.real01 rng
      val offset = u0 / real n
      val old = Array.tabulate (n, fn i => Array.sub (parts, i))
      fun findK k thresh =
        if k >= n - 1 orelse Array.sub (cumW, k) >= thresh then k
        else findK (k + 1) thresh
      fun resample i k =
        if i >= n then ()
        else
          let val thresh = offset + real i / real n
              val k' = findK k thresh
          in Array.update (parts, i, Array.sub (old, k')); resample (i + 1) k' end
      val () = resample 0 0
      val resampled = List.tabulate (n, fn i => Array.sub (parts, i))
    in
      ({ particles = resampled, weights = wList, mean = mean, ess = ess }, rng')
    end

  (* ---- runners ---- *)

  fun initParticles (n, x0, initVar) seed =
    let
      val rng = ref (Prng.SplitMix64.seed seed)
      val sd = Math.sqrt initVar
      val parts = Array.tabulate (n, fn _ =>
        let val (g, rng') = gauss1 (!rng)
        in rng := rng'; x0 + g * sd end)
    in (parts, !rng) end

  fun runSIRTrace ({ n, q, r, initVar } : params) obs x0 seed =
    let
      val (parts, rng0) = initParticles (n, x0, initVar) seed
      fun go ([], _, acc) = List.rev acc
        | go (z :: zs, rng, acc) =
            let val (st, rng') = sirOneStep (q, r) parts z rng
            in go (zs, rng', st :: acc) end
    in go (obs, rng0, []) end

  fun runSIRWith params obs x0 seed =
    case List.rev (runSIRTrace params obs x0 seed) of
        [] => x0
      | last :: _ => #mean last

  fun runSIR obs x0 seed = runSIRWith defaultParams obs x0 seed

  fun runKalman obs x0 =
    let
      val { q, r, ... } = defaultParams
      val f = Matrix.fromRows [[1.0]]
      val b = Matrix.fromRows [[0.0]]
      val qm = Matrix.fromRows [[q]]
      val h = Matrix.fromRows [[1.0]]
      val rm = Matrix.fromRows [[r]]
      val model = { f = f, b = b, q = qm, h = h, r = rm }
      val st0 = { x = Matrix.fromRows [[x0]], p = Matrix.fromRows [[1.0]] }
      val u   = Matrix.fromRows [[0.0]]
      val stN = List.foldl
                  (fn (z, st) =>
                    Kalman.step model { u = u, z = Matrix.fromRows [[z]] } st)
                  st0 obs
    in Matrix.sub (#x stN, 0, 0) end

  fun filterStep z x = runSIR [z] x 0wx42

  fun compareKalman () =
    let
      val obs = [1.0, 1.0, 1.0, 1.0, 1.0]
      val k = runKalman obs 0.0
      val p = runSIR   obs 0.0 0wx1234
    in (k, p) end
end

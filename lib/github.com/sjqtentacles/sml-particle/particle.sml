(* SIR particle filter + Kalman comparison for 1-D random-walk tracking.

   Model:   x_{t+1} = x_t + w,   w ~ N(0, Q)   (process)
            z_t     = x_t + v,   v ~ N(0, R)   (measurement)

   Q = 0.01, R = 0.1, x_0 = 0.0.
   `compareKalman` runs both filters on observations [1.0,1.0,1.0,1.0,1.0]
   and returns (kalman_estimate, particle_estimate). *)

structure Particle :> PARTICLE =
struct
  val nParticles = 200
  val q = 0.01   (* process noise variance *)
  val r = 0.1    (* observation noise variance *)

  (* Box-Muller: one standard Gaussian from two U[0,1) samples *)
  fun gauss1 rng =
    let
      val (u1, rng1) = Prng.SplitMix64.real01 rng
      val (u2, rng2) = Prng.SplitMix64.real01 rng1
      val mag = Math.sqrt (~2.0 * Math.ln (Real.max (u1, 1E~300)))
    in (mag * Math.cos (2.0 * Math.pi * u2), rng2) end

  (* SIR step: predict + weight + resample.
     Mutates `parts` in-place; returns updated rng. *)
  fun sirOneStep parts z rng =
    let
      val n = Array.length parts

      (* Predict: add process noise *)
      val rng =
        Array.foldli (fn (i, x, rng) =>
          let val (g, rng') = gauss1 rng
          in Array.update (parts, i, x + g * Math.sqrt q); rng' end)
          rng parts

      (* Log-weights: log N(z | x_i, R) = -0.5*(z-x)^2/R + const *)
      val logW = Array.tabulate (n, fn i =>
        ~0.5 * (z - Array.sub (parts, i)) * (z - Array.sub (parts, i)) / r)

      (* Shift by max for numerical stability, then exponentiate *)
      val maxLW = Array.foldl Real.max (Array.sub (logW, 0)) logW
      val w = Array.tabulate (n, fn i => Math.exp (Array.sub (logW, i) - maxLW))
      val tot = Array.foldl op+ 0.0 w
      val () = Array.modify (fn wi => wi / tot) w

      (* Cumulative weights for systematic resampling *)
      val cumW = Array.array (n, 0.0)
      val () = Array.update (cumW, 0, Array.sub (w, 0))
      val () = List.app (fn i =>
        Array.update (cumW, i, Array.sub (cumW, i - 1) + Array.sub (w, i)))
        (List.tabulate (n - 1, fn k => k + 1))

      (* Systematic resampling *)
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
    in rng' end

  (* Run SIR for a sequence of observations; returns final mean *)
  fun runSIR obs x0 seed =
    let
      val n = nParticles
      val rng = ref (Prng.SplitMix64.seed seed)
      (* Initialise particles as N(x0, 1.0) *)
      val parts = Array.tabulate (n, fn _ =>
        let val (g, rng') = gauss1 (!rng)
        in rng := rng'; x0 + g end)
      val () = List.app (fn z => rng := sirOneStep parts z (!rng)) obs
    in Array.foldl op+ 0.0 parts / real n end

  (* Run Kalman filter for a sequence of observations; returns final x estimate *)
  fun runKalman obs x0 =
    let
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

  (* filterStep z x: one-shot SIR step: N particles initialised near x,
     observe z, return weighted mean. *)
  fun filterStep z x =
    runSIR [z] x 0wx42

  (* compareKalman: run both filters on obs=[1,1,1,1,1] from x0=0.
     True state is 1.0; both estimators should converge near 1.0. *)
  fun compareKalman () =
    let
      val obs = [1.0, 1.0, 1.0, 1.0, 1.0]
      val k = runKalman obs 0.0
      val p = runSIR   obs 0.0 0wx1234
    in (k, p) end
end

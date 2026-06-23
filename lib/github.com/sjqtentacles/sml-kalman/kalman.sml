(* kalman.sml -- linear Kalman filter + recursive least squares over sml-matrix.

   Everything is expressed with the dense real-matrix operations of the vendored
   `structure Matrix`: `mul`, `add`, `sub'`, `transpose`, `scale`, `inv`,
   `identity`, and the scalar accessor `sub`. Column vectors are n x 1
   matrices. The implementation is a direct transcription of the standard
   Kalman recursions, so it is deterministic and identical on both compilers. *)

structure Kalman :> KALMAN =
struct
  structure M = Matrix

  type state = { x : M.t, p : M.t }
  type model = { f : M.t, b : M.t, q : M.t, h : M.t, r : M.t }

  fun colVec xs = M.fromRows (List.map (fn x => [x]) xs)
  fun toList v = List.map (fn row => hd row) (M.toRows v)

  fun predict ({ f, b, q, ... } : model) u ({ x, p } : state) : state =
    let
      val x' = M.add (M.mul (f, x), M.mul (b, u))
      val p' = M.add (M.mul (M.mul (f, p), M.transpose f), q)
    in { x = x', p = p' } end

  fun update ({ h, r, ... } : model) z ({ x, p } : state) : state =
    let
      val ht = M.transpose h
      val y = M.sub' (z, M.mul (h, x))                  (* innovation *)
      val s = M.add (M.mul (M.mul (h, p), ht), r)       (* innovation cov *)
      val k = M.mul (M.mul (p, ht), M.inv s)            (* Kalman gain *)
      val x' = M.add (x, M.mul (k, y))
      val n = M.rows p
      val ikh = M.sub' (M.identity n, M.mul (k, h))
      val p' = M.mul (ikh, p)
    in { x = x', p = p' } end

  fun step (m : model) { u, z } st = update m z (predict m u st)

  structure RLS =
  struct
    type estimator = { theta : M.t, p : M.t }

    fun init n delta = { theta = M.zeros (n, 1), p = M.scale delta (M.identity n) }

    fun update lambda ({ theta, p } : estimator) { phi, y } : estimator =
      let
        val phiT = M.transpose phi                       (* 1 x n *)
        val pphi = M.mul (p, phi)                         (* n x 1 *)
        val denom = lambda + M.sub (M.mul (phiT, pphi), 0, 0)
        val k = M.scale (1.0 / denom) pphi                (* gain, n x 1 *)
        val pred = M.sub (M.mul (phiT, theta), 0, 0)
        val err = y - pred
        val theta' = M.add (theta, M.scale err k)
        val p' = M.scale (1.0 / lambda)
                   (M.sub' (p, M.mul (k, M.mul (phiT, p))))
      in { theta = theta', p = p' } end

    fun coeffs ({ theta, ... } : estimator) =
      List.map (fn row => hd row) (M.toRows theta)
  end
end

(* kalman.sig

   The linear (discrete-time) Kalman filter and a recursive least-squares
   estimator, in pure Standard ML on top of the vendored `sml-matrix`
   (`structure Matrix`). All states, covariances, and model matrices are dense
   `Matrix.t` values; vectors are n x 1 column matrices.

   The state-space model is

       x_k = F x_{k-1} + B u_k + w,   w ~ (0, Q)      (process)
       z_k = H x_k       + v,         v ~ (0, R)      (measurement)

   `predict` advances the state estimate and its covariance through the process
   model; `update` folds in a measurement via the Kalman gain. `step` does a
   predict immediately followed by an update.

   The recursive least-squares (`RLS`) substructure tracks the coefficient
   vector of a linear model y = phi . theta one sample at a time, with an
   optional forgetting factor lambda. *)

signature KALMAN =
sig
  (* A filter state: the estimate `x` (n x 1) and its covariance `p` (n x n). *)
  type state = { x : Matrix.t, p : Matrix.t }

  (* A linear-Gaussian model. f : n x n, b : n x p, q : n x n,
     h : m x n, r : m x m. *)
  type model =
    { f : Matrix.t, b : Matrix.t, q : Matrix.t
    , h : Matrix.t, r : Matrix.t }

  (* Column-vector helpers. *)
  val colVec : real list -> Matrix.t      (* n reals -> n x 1 matrix *)
  val toList : Matrix.t -> real list       (* n x 1 matrix -> n reals *)

  (* Predict step: x <- F x + B u ;  P <- F P F^T + Q. `u` is the control input
     (p x 1); pass a zero vector when there is no control. *)
  val predict : model -> Matrix.t -> state -> state

  (* Update step with measurement z (m x 1):
       y = z - H x ;  S = H P H^T + R ;  K = P H^T S^-1 ;
       x <- x + K y ;  P <- (I - K H) P. *)
  val update : model -> Matrix.t -> state -> state

  (* One predict + update with control u and measurement z. *)
  val step : model -> { u : Matrix.t, z : Matrix.t } -> state -> state

  structure RLS :
  sig
    (* theta : n x 1 coefficient estimate; p : n x n inverse-correlation. *)
    type estimator = { theta : Matrix.t, p : Matrix.t }

    (* init n delta : zero coefficients, P = delta * I (delta large => weak
       prior). *)
    val init : int -> real -> estimator

    (* update lambda est { phi, y } : fold in one (regressor, response) pair
       with forgetting factor lambda (use 1.0 for ordinary RLS). *)
    val update : real -> estimator -> { phi : Matrix.t, y : real } -> estimator

    val coeffs : estimator -> real list
  end
end

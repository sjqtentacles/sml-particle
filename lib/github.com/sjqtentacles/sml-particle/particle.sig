signature PARTICLE =
sig
  val filterStep : real -> real -> real
  val compareKalman : unit -> real * real
end

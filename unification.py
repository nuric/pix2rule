"""Subgraph isomorphism based unification."""
import tensorflow as tf

from utils.ops import reduce_probsum


def unify(
    inv_unary: tf.Tensor,
    inv_binary: tf.Tensor,
    batch_unary: tf.Tensor,
    batch_binary: tf.Tensor,
    iterations: int = 1,
) -> tf.Tensor:
    """Unify given rules with a batch of examples."""
    # inv_unary (I, N, P1)
    # inv_binary (I, N, N, P2)
    # batch_unary (B, M, P1)
    # batch_binary (B, M, M, P2)
    # ---------------------------
    # Match unary conditions to form the initial assignment sets
    uni_sets = (
        inv_unary[:, :, None] * batch_unary[:, None, None] + 1 - inv_unary[:, :, None]
    )  # (B, I, N, M, P1)
    # The following tensor tells us which M elements are possible assignments to N variables
    # the reduce_prod models the conjunction of unary predicates
    uni_sets = tf.reduce_prod(uni_sets, -1)  # (B, I, N, M)
    # ---------------------------
    # The following tensor finds possible paring matches,
    # for every pair in invariants, find the match for every other pair
    pair_match = inv_binary[..., None, None, :] * batch_binary[:, None, None, None] + (
        1 - inv_binary[..., None, None, :]
    )
    # (B, I, N, N, M, M, P2)
    pair_match = tf.reduce_prod(pair_match, -1)  # (B, I, N, N, M, M)
    # pair_match *= (1 - tf.eye(self.inv_inputs.shape[1])[..., None, None]) * (
    # 1 - tf.eye(inputs.shape[1])
    # )
    # pair_match *= 1 - tf.eye(batch_unary.shape[1])  # rule out self relations p(X,X)
    # ---------------------------
    # Iterate over binary predicates
    for _ in range(iterations):
        # (B, I, N, M) x  (B, I, N, N, M, M) -> (B, I, N, N, M, M)
        reduct = tf.einsum("bimk,binj,bimnkj->bimnkj", uni_sets, uni_sets, pair_match)
        # (B, I, N, M)
        lr_reduct = tf.reduce_prod(reduce_probsum(reduct, 4), 2)
        # (B, I, N, M)
        rl_reduct = tf.reduce_prod(reduce_probsum(reduct, 5), 3)
        uni_sets = lr_reduct * rl_reduct  # (B, I, N, M)
    # ---------------------------
    return uni_sets

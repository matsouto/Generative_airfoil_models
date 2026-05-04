import tensorflow as tf
import numpy as np


def compute_vod_numpy(coords):
    """
    Calculates VOD for a single airfoil (Numpy array [N, 2]).
    Focuses on Y coordinates.
    """
    y = coords[:, 1]  # Extract Y

    # 1. First Difference (Slope)
    diffs = np.diff(y)

    # 2. Variance of Differences (Smoothness)
    vod = np.var(diffs)

    return vod


def compute_vod_loss(y_coords):
    """
    Calcula a perda baseada na Variação da Diferença (VOD).
    Fórmula: VOD = Variância(y_{i+1} - y_i)

    Args:
        y_coords: Tensor de shape (Batch, N_Points) contendo apenas as coordenadas Y.
    Returns:
        Um valor escalar (média do VOD do batch).
    """
    # 1. Primeira Diferença (Slopes / Inclinações)
    # Equivalente a y[i+1] - y[i]
    # Shape resultante: (Batch, N_Points - 1)
    diffs = y_coords[:, 1:] - y_coords[:, :-1]

    # 2. Variância das Diferenças
    # Calcula quão inconsistente é a inclinação ao longo do perfil.
    # Se a linha for reta ou uma curva perfeita constante, a variância é baixa.
    # Se for um ziguezague, a variância é alta.
    # axis=1 calcula a variância ao longo dos pontos do perfil (não do batch)
    vod = tf.math.reduce_variance(diffs, axis=1)

    # 3. Média sobre o Batch
    # Retorna um único número para somar à Loss total
    return tf.reduce_mean(vod)

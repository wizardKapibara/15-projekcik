"""Porównanie obrazów przez dopasowanie punktów kluczowych ORB."""

from __future__ import annotations

import cv2
import numpy as np


def orb_score(img1: np.ndarray, img2: np.ndarray, n_features: int = 500) -> float:
    """Oblicz wynik podobieństwa ORB między dwoma obrazami.

    Używa deskryptora ORB (Oriented FAST + Rotated BRIEF) i matchera BFMatcher.
    Score = liczba dobrych dopasowań / max(kp1, kp2). Wyższy = bardziej podobne.

    Args:
        img1, img2: obrazy grayscale lub RGB.
        n_features: maksymalna liczba wykrywanych punktów kluczowych.

    Returns:
        Score ∈ [0, 1].
    """
    def _gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    g1 = _gray(img1)
    g2 = _gray(img2)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    # Filtruj dobre dopasowania (Lowe-style ratio test z crossCheck)
    matches = sorted(matches, key=lambda m: m.distance)
    good_matches = [m for m in matches if m.distance < 64]

    score = len(good_matches) / max(len(kp1), len(kp2))
    return min(float(score), 1.0)


def orb_keypoints_image(img: np.ndarray, n_features: int = 500) -> tuple[np.ndarray, int]:
    """Narysuj punkty kluczowe ORB na obrazie (do wizualizacji).

    Returns:
        (image_with_keypoints_rgb, n_keypoints)
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        vis = img.copy()
    else:
        gray = img
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp, _ = orb.detectAndCompute(gray, None)
    cv2.drawKeypoints(vis, kp, vis, color=(0, 200, 0),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return vis, len(kp)

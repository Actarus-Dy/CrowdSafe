# C-14 · CrowdSafe
## Simulation et sécurité des foules par équations gravitationnelles Janus
### Plan technique détaillé — Extension civile du Modèle Cosmologique de Janus

> **Mars 2026** · Claude Sonnet 4.6 · Basé sur le corpus Janus (ArXiv 2412.04644 · Part-1 & Part-2)

---

## Fiche synthétique

| Indicateur | Valeur |
|---|---|
| Identifiant | C-14 |
| Priorité | Haute |
| TRL actuel / cible | 4 / 7 |
| Délai estimé | 3 à 5 mois |
| Budget estimé | 50 000 € |
| Technologies | Python 3.12, C++ (libcrowd), Mesa ABM, SciPy, FastAPI, WebSocket, React |
| Domaines mathématiques | **§1.1–1.3 (N-corps) · §4.3 (Christoffel) · §4.4 (Géodésiques) · §5.5 (TOV) · §5.6 (Schwarzschild)** |
| Applications | Stades, concerts, gares, évacuations d'urgence, fêtes populaires, pèlerinages |
| Impact public | 5/5 — Très élevé · Sécurité des personnes |

---

## Idée centrale — La foule comme univers N-corps avec pression stellaire

CrowdSafe est le logiciel dont l'isomorphie est la plus immédiatement intuitive de toute la série : **une foule dense est un gaz de particules gravitationnelles avec une pression interne** — exactement comme le cœur d'une étoile. L'équation TOV (§5.5) qui décrit comment la pression stellaire pousse vers l'extérieur (évacuation) ou vers l'intérieur (effondrement) décrit aussi comment la pression de foule s'accumule dans un espace confiné. Le rayon de Schwarzschild (§5.6) devient la **densité critique de foule** au-delà de laquelle le mouvement est irréversible — c'est le seuil physiologique de 6 personnes/m² au-delà duquel les blessures graves deviennent inévitables.

---

## Table des matières

1. [Fondements mathématiques — Quadruple domaine](#1-fondements-mathématiques)
2. [Architecture logicielle](#2-architecture-logicielle)
3. [Spécification des algorithmes clés](#3-spécification-des-algorithmes-clés)
4. [Spécification API](#4-spécification-api)
5. [Feuille de route — 5 mois](#5-feuille-de-route--5-mois)
6. [Analyse des risques](#6-analyse-des-risques)
7. [Budget et indicateurs de succès](#7-budget-et-indicateurs-de-succès)

---

## 1. Fondements mathématiques

### 1.1 Table d'analogie complète — §1 + §4 + §5 ↔ Dynamique des foules

| Grandeur Janus | Grandeur CrowdSafe | Signification sécurité |
|---|---|---|
| Masse positive ρ⁺ (§1.1) | Piéton marchant librement | Se déplace vers sa destination |
| Masse négative ρ⁻ (§1.2) | Obstacle, mur, foule très dense | Repousse les piétons (évitement) |
| Attraction (+,+) §1.2 | Comportement grégaire (suivi de foule) | Piétons s'attirent mutuellement |
| Répulsion (+,−) §1.2 | Évitement d'obstacles | Piéton fuit murs et zones denses |
| Potentiel Φ §1.3 | Champ de densité de foule | Carte de pression et de risque |
| TOV dp/dr §5.5 | Gradient de pression de foule dP/dl | Accumulation de pression dans un couloir |
| Rayon de Schwarzschild §5.6 | Densité critique ρ_c = 6 pers/m² | Seuil de danger physiologique irréversible |
| Géodésique §4.4 | Chemin d'évacuation optimal | Trajet le plus court vers la sortie |
| Christoffel §4.3 | Gradient de densité locale | Réfraction du trajet d'évacuation |
| Singularité de courbure | Zone de compression mortelle | Piège à foule — alerte maximale |

---

### 1.2 Masses signées des piétons — §1.2

$$m_{\text{piéton}} = \begin{cases} +(v - v_{\text{cible}}) & \text{si piéton mobile, dans sa direction} \\ -(|v_{\text{obstacle}}|) & \text{si obstacle statique ou zone surpeuplée} \end{cases}$$

**Règles d'interaction §1.2 — émergences spontanées :**

| Interaction | Règle §1.2 | Phénomène observé |
|---|---|---|
| Piéton(+) ↔ Piéton(+) | Attraction faible | Marche en groupe, comportement grégaire |
| Piéton(+) ↔ Obstacle(−) | Répulsion | Évitement de murs, contournement obstacles |
| Piéton(+) ↔ Foule dense(−) | Répulsion forte | Tentative de fuite — si impossible → compression |
| Foule dense(−) ↔ Foule dense(−) | Attraction | Agglomération → catastrophe si espace fermé |

---

### 1.3 TOV — Pression de foule dans un espace confiné (§5.5)

$$\frac{dp}{dr} = -\frac{(\rho c^2 + p)(m + 4\pi r^3 p/c^2)}{r(r-2m)c^2} \quad (\S5.5) \xrightarrow{c\to\infty} \frac{dP_{\text{foule}}}{dl} = -\rho_{\text{foule}}(l) \cdot F_{\text{contact}}(l)$$

- $P_{\text{foule}}$ = pression de contact entre personnes [N/m]
- $\rho_{\text{foule}}$ = densité de personnes [pers/m²]
- $F_{\text{contact}}$ = force de contact inter-individus ≈ 100 N/pers
- $l$ = distance le long du couloir ou de la sortie

**Application :** L'intégration TOV depuis l'entrée d'un espace vers la sortie donne le profil de pression de foule. Quand la pression dépasse 4 450 N/m (seuil physiologique d'écrasement pulmonaire), l'alerte est déclenchée automatiquement.

---

### 1.4 Densité critique — Schwarzschild (§5.6)

$$r_s = \frac{2GM}{c^2} \quad (\S5.6) \quad \longleftrightarrow \quad \rho_c = 6 \text{ pers/m}^2$$

Le rayon de Schwarzschild est le seuil en dessous duquel une étoile s'effondre irrémédiablement. La densité critique de foule $\rho_c = 6$ pers/m² est le seuil analogue : en dessous, les individus ont encore un degré de liberté de mouvement. Au-delà, le mouvement devient **collectif et incontrôlable** — les individus ne peuvent plus choisir leur direction.

```
ρ < 2 pers/m²    → circulation libre (marche normale)
ρ ∈ [2, 4] pers/m² → circulation contrainte (file lente)
ρ ∈ [4, 6] pers/m² → surveillance active (alerte orange)
ρ > 6 pers/m²    → DANGER (analogue r < r_s) → évacuation immédiate
ρ > 8 pers/m²    → CRITIQUE : pression d'écrasement → urgence
```

---

### 1.5 Géodésique d'évacuation — §4.4

$$\frac{d^2 x^\mu}{d\tau^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\tau} \frac{dx^\beta}{d\tau} = 0 \quad (\S4.4)$$

**Métrique de densité (§4.5) :** La vitesse de marche dépend de la densité locale.

$$g_{\mu\nu}(x,y) = \frac{1}{v_{\text{marche}}^2(\rho(x,y))} \cdot \delta_{\mu\nu}$$

$$v_{\text{marche}}(\rho) = v_0 \cdot \max\left(0, 1 - \frac{\rho}{\rho_c}\right) \quad (\text{modèle Weidmann})$$

Les Christoffel (§4.3) encodent le gradient de densité :

$$\Gamma^\mu_{\alpha\beta} \sim -\partial_\mu(\ln v_{\text{marche}}) \sim +\partial_\mu \rho \quad \text{→ réfraction vers les zones moins denses}$$

L'algorithme RK4 §4.4 (réutilisé depuis C-02, C-09, C-10, C-12) calcule le chemin d'évacuation optimal depuis chaque position vers la sortie la plus proche, en tenant compte des zones de densité maximale à éviter.

---

## 2. Architecture logicielle

```
┌────────────────────────────────────────────────────────────────┐
│  COUCHE 7 — Dashboard opérateur & Sécurité (React · Leaflet)    │
│  Carte densité temps réel · Alertes push · Plan évacuation 3D   │
└────────────────────────┬───────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────┐
│  COUCHE 6 — Moteur de foule Janus (Python/C++)  ★ CŒUR ★        │
│  CrowdNBody(§1) · TOVPressure(§5.5) · CritDensity(§5.6)        │
│  EvacGeodesic(§4.4) · DensityPotential(§1.3) · AlarmSystem     │
└────────────────────────┬───────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────┐
│  COUCHE 5 — Acquisition capteurs (MQTT · Computer Vision)        │
│  Caméras comptage (YOLO v9) · Capteurs pression au sol          │
│  Bracelet RFID / NFC · Données billetterie                       │
└────────────────────────┬───────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────┐
│  COUCHE 4 — Stockage & Streaming                                │
│  InfluxDB (mesures 1Hz) · PostGIS (géométrie lieu) · Redis      │
└────────────────────────────────────────────────────────────────┘
```

### Composants du moteur Janus (Couche 6)

| Module | Source §Janus | Rôle |
|---|---|---|
| `CrowdNBody` | §1.1 · §1.2 | Dynamique N-corps des piétons |
| `DensityPotential` | §1.3 | Carte de pression Φ(x,y) temps réel |
| `TOVPressure` | §5.5 | Profil de pression dans les couloirs |
| `CriticalDensity` | §5.6 | Détection seuil 6 pers/m² |
| `EvacGeodesic` | §4.4 · §4.3 | Calcul des chemins d'évacuation optimaux |
| `AlarmSystem` | §5.5 · §5.6 | Déclenchement alertes par seuillage |

---

## 3. Spécification des algorithmes clés

### 3.1 N-corps piétons — Modèle de force sociale Janus (§1.1–1.3)

```python
import mesa
import numpy as np

G_SOCIAL  = 2.0     # Constante d'attraction sociale [m²/s²/pers]
G_REPULSE = 5.0     # Constante de répulsion obstacle [m²/s²]
R_PERSONAL = 0.5    # Espace personnel [m] — distance de confort

class PedestrianAgent(mesa.Agent):
    """
    Piéton modélisé comme particule gravitationnelle §1.

    Masse signée §1.2 :
      m = +1.0  → piéton mobile (matière positive)
      m = −inf  → obstacle statique (mur, poteau)
      m = −ρ_local  → zone surpeuplée (masse négative proportionnelle à la densité)

    Forces §1.1 :
      F_attracteur  = G_social  × mᵢ × mⱼ / dᵢⱼ²  (suivi de groupe)
      F_répulseur   = G_repulse × mᵢ × |mⱼ| / dᵢⱼ²  (évitement)
      F_destination = vecteur directeur vers sortie (§4.4 géodésique)
    """

    def __init__(self, uid, model, pos: np.ndarray,
                 destination: np.ndarray, mass: float = 1.0):
        super().__init__(uid, model)
        self.pos         = pos.copy()
        self.vel         = np.zeros(2)
        self.destination = destination
        self.mass        = mass    # §1.2 : +1 piéton, −ρ obstacle

    def step(self):
        # === Force 1 : Destination (géodésique §4.4) ===
        d_to_dest    = self.destination - self.pos
        dist_dest    = np.linalg.norm(d_to_dest) + 1e-6
        v_desired    = self.model.v_max * d_to_dest / dist_dest
        F_dest       = (v_desired - self.vel) * 0.5   # relâchement

        # === Force 2 : N-corps §1.1 (Barnes-Hut 2D) ===
        F_social = self.model.barnes_hut.compute_force(self)

        # === Force 3 : Répulsion obstacles §1.2 ===
        F_obstacle = np.zeros(2)
        for obs in self.model.obstacles:
            d = self.pos - obs.pos
            r = np.linalg.norm(d) + 1e-6
            if r < 3.0:   # rayon d'influence obstacle [m]
                F_obstacle += G_REPULSE * d / r**3

        # === Intégration §1 (leapfrog symplectique) ===
        F_total   = F_dest + F_social + F_obstacle
        self.vel += F_total * self.model.dt
        # Vitesse maximale dépend de la densité locale §5.6
        v_max_local = self.model.v_max * (1 - self.model.local_density(self.pos) / 6.0)
        v_norm      = np.linalg.norm(self.vel) + 1e-6
        if v_norm > max(v_max_local, 0.1):
            self.vel = self.vel / v_norm * max(v_max_local, 0.1)
        self.pos += self.vel * self.model.dt
```

**Émergences spontanées §1.2 (sans paramétrisation empirique) :**
- Comportement de lane-formation (colonnes spontanées)
- Arching devant les sorties (voûtes de pression)
- Phénomène "faster-is-slower" (panique → blocage)
- Stop-and-go waves dans les couloirs étroits

---

### 3.2 TOV — Profil de pression dans les couloirs (§5.5)

```python
def tov_crowd_pressure(l: np.ndarray,
                        rho_profile: np.ndarray,
                        width_m: float,
                        F_contact: float = 100.0) -> dict:
    """
    Profil de pression de foule — transposition directe de TOV §5.5.

    Étoile §5.5 :  dp/dr = −ρ·g_eff·(1 + corrections relativistes)
    Foule        :  dP/dl = −ρ_foule(l) · F_contact

    P(l) = pression de contact [N/m] le long du couloir
    ρ(l) = densité de personnes [pers/m²]
    F_contact = 100 N/pers (force typique en foule dense)
    width_m   = largeur du couloir [m]

    Seuil d'alerte (analogue r_s §5.6) :
      P > 4 450 N/m → risque d'écrasement thoracique → ÉVACUATION
    """
    from scipy.integrate import cumulative_trapezoid

    # Pression cumulée le long du couloir (intégrale de TOV §5.5)
    dP_dl = -rho_profile * F_contact * width_m    # [N/m par mètre]
    P     = cumulative_trapezoid(-dP_dl, l, initial=0)

    # Seuils de sécurité (analogues aux seuils stellaires §5.5)
    P_max       = np.max(np.abs(P))
    DANGER_N_m  = 4450.0   # seuil physiologique [N/m]
    WARNING_N_m = 2700.0   # seuil de surveillance

    return {
        'l_m':            l,
        'pressure_N_m':   P,
        'P_max':          float(P_max),
        'alert_level':    ('ROUGE'   if P_max > DANGER_N_m  else
                           'ORANGE'  if P_max > WARNING_N_m else 'VERT'),
        'location_max_m': float(l[np.argmax(np.abs(P))]),
        'schwarzschild_analogy': {
            'r_s_equiv': P_max / DANGER_N_m,   # > 1 → danger §5.6
            'critical_exceeded': P_max > DANGER_N_m
        }
    }
```

---

### 3.3 Densité critique — Schwarzschild §5.6

```python
def critical_density_monitor(density_map: np.ndarray,
                               dx_m: float = 0.5) -> dict:
    """
    Surveillance temps réel de la densité critique §5.6.

    Schwarzschild §5.6 : r_s = 2GM/c² → seuil d'effondrement irréversible
    CrowdSafe           : ρ_c = 6 pers/m² → seuil de danger irréversible

    Au-delà de ρ_c, les individus ne peuvent plus choisir leur direction :
      le mouvement devient collectif et incontrôlable (analogue de r < r_s).
    Cette zone est signalée immédiatement comme zone de piège à foule.
    """
    RHO_C_CRITICAL = 6.0   # pers/m² — seuil Schwarzschild §5.6
    RHO_C_DANGER   = 4.0   # pers/m² — alerte orange
    RHO_C_WATCH    = 2.5   # pers/m² — surveillance

    n_critical = np.sum(density_map > RHO_C_CRITICAL)
    n_danger   = np.sum(density_map > RHO_C_DANGER)
    area_pixel = dx_m**2

    # Zones critiques : analogues aux régions sous r_s (Schwarzschild §5.6)
    critical_mask = density_map > RHO_C_CRITICAL

    return {
        'max_density':       float(density_map.max()),
        'mean_density':      float(density_map.mean()),
        'critical_area_m2':  float(n_critical * area_pixel),
        'danger_area_m2':    float(n_danger   * area_pixel),
        'alert_level':       ('ROUGE'  if n_critical > 0 else
                              'ORANGE' if n_danger   > 0 else 'VERT'),
        'critical_mask':     critical_mask,   # carte des zones § r_s
        'compacity_ratio':   float(density_map.max() / RHO_C_CRITICAL),
        'schwarzschild_exceeded': bool(n_critical > 0)
    }
```

---

### 3.4 Géodésique d'évacuation — §4.4

```python
def evacuation_geodesic(start: np.ndarray,
                          exits: list,
                          density_map: np.ndarray,
                          dx_m: float = 0.5,
                          v_max: float = 1.5) -> dict:
    """
    Calcul du chemin d'évacuation optimal par géodésique §4.4.

    L'eau, la lumière et les piétons en évacuation suivent tous
    le même principe : le chemin de moindre résistance = géodésique.

    Métrique §4.5 : g_uv = 1/v²(ρ)·δ_uv
    v(ρ) = v_max · max(0, 1 − ρ/ρ_c)   (Weidmann)

    Christoffel §4.3 : Γ ~ ∂ρ  (gradient de densité = réfraction)

    Algorithme : Dijkstra pondéré sur la grille de densité
    (approximation discrète de l'intégration RK4 §4.4 continu)
    Réutilise le code de C-02, C-09, C-10, C-12.
    """
    import heapq

    ny, nx = density_map.shape
    RHO_C  = 6.0

    # Métrique §4.5 : temps de traversée de chaque cellule
    # = dx_m / v(ρ)  →  ∞ si ρ > ρ_c (zone infranchissable §5.6)
    v_map     = v_max * np.maximum(0, 1 - density_map / RHO_C)
    cost_map  = np.where(v_map > 0.01, dx_m / v_map, np.inf)

    # Dijkstra depuis tous les exits simultanément (multi-source)
    dist = np.full((ny, nx), np.inf)
    heap = []
    for ex in exits:
        ix, iy = int(ex[0]/dx_m), int(ex[1]/dx_m)
        if 0 <= ix < nx and 0 <= iy < ny:
            dist[iy, ix] = 0.0
            heapq.heappush(heap, (0.0, ix, iy))

    while heap:
        d, ix, iy = heapq.heappop(heap)
        if d > dist[iy, ix]:
            continue
        for dix, diy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
            nx_, ny_ = ix+dix, iy+diy
            if 0 <= nx_ < nx and 0 <= ny_ < ny:
                step = dx_m * (1.414 if abs(dix)+abs(diy)==2 else 1.0)
                nd   = d + cost_map[ny_, nx_] * step / dx_m
                if nd < dist[ny_, nx_]:
                    dist[ny_, nx_] = nd
                    heapq.heappush(heap, (nd, nx_, ny_))

    # Trajet depuis le point de départ (gradient descent sur dist)
    sx, sy = int(start[0]/dx_m), int(start[1]/dx_m)
    path   = [start.copy()]
    x, y   = sx, sy

    for _ in range(10000):
        neighbors = [(x+dix, y+diy) for dix,diy in
                     [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
                     if 0 <= x+dix < nx and 0 <= y+diy < ny]
        if not neighbors:
            break
        best = min(neighbors, key=lambda p: dist[p[1], p[0]])
        if dist[best[1], best[0]] >= dist[y, x]:
            break
        x, y = best
        path.append(np.array([x * dx_m, y * dx_m]))
        if dist[y, x] < 0.01:
            break

    travel_time = float(dist[sy, sx])
    return {
        'path':            np.array(path),
        'travel_time_s':   travel_time,
        'path_length_m':   len(path) * dx_m,
        'bottleneck_rho':  float(density_map[sy, sx]),
        'evac_feasible':   travel_time < np.inf,
        'distance_map':    dist    # carte des temps d'évacuation pour tout le venue
    }
```

---

### 3.5 Potentiel de pression Φ et carte de risque (§1.3)

```python
def crowd_pressure_map(agents: list,
                         grid: np.ndarray,
                         G_press: float = 2.0) -> np.ndarray:
    """
    Carte du potentiel de pression de foule Φ(x,y) — §1.3.

    §1.3 Janus : Φ±(r) = ∓ G|M| / r
    CrowdSafe  : Φ(x,y) = Σᵢ −G_press · ρᵢ / rᵢ   (pression convergente)
                         + Σⱼ +G_press · 1/rⱼ   (sorties = puits de sécurité)

    Φ < 0 → zone de pression élevée → risque de compression
    Φ > 0 → zone libérée (près des sorties)
    |∇Φ|   → gradient de pression → direction de la force sur les piétons

    Identique à C-01, C-04, C-12, C-13 §1.3 — code partageable.
    """
    phi = np.zeros(grid.shape[:2])

    # Contribution des piétons : pression convergente
    for a in agents:
        if a.mass > 0:    # piéton = masse positive
            r = np.sqrt((grid[:,:,0] - a.pos[0])**2 +
                        (grid[:,:,1] - a.pos[1])**2) + 0.3  # 30cm min
            phi -= G_press * a.mass / r

    # Contribution des sorties : puits de sécurité (masse négative §1.2)
    for exit_pos in grid['exits']:
        r = np.sqrt((grid[:,:,0] - exit_pos[0])**2 +
                    (grid[:,:,1] - exit_pos[1])**2) + 0.5
        phi += G_press * 10.0 / r   # sorties = attracteurs forts (−masse)

    return phi
```

---

## 4. Spécification API

**Base URL :** `https://api.crowdsafe.io/api/v1`

### 4.1 Endpoints REST

| Méthode | Endpoint | Description | Latence P95 |
|---|---|---|---|
| `POST` | `/venue/load` | Chargement du plan de salle (DXF/GeoJSON) | 2 s |
| `GET` | `/density/live/{venue_id}` | Densité temps réel par zone | 100 ms |
| `GET` | `/pressure/tov/{corridor_id}` | Profil TOV §5.5 d'un couloir | 200 ms |
| `POST` | `/evacuation/plan` | Calcul géodésique §4.4 depuis position | 500 ms |
| `POST` | `/simulate/scenario` | Simulation N-corps sur 30 min | 30 s |
| `GET` | `/risk-map/{venue_id}` | Carte Φ §1.3 risque de compression | 1 s |
| `WS` | `/stream/{venue_id}` | Alertes densité temps réel · 2 Hz | < 100 ms |

### 4.2 WS `/stream/{venue_id}` — Alertes temps réel

```
Server → Client (2 Hz) :
{
  "type":           "density_update",
  "timestamp":      "ISO8601",
  "zones": [
    { "id": "A1", "density": 3.2, "alert": "VERT",
      "P_TOV_N_m": 1840, "schwarzschild_ratio": 0.53 },
    { "id": "B3", "density": 5.8, "alert": "ORANGE",
      "P_TOV_N_m": 3920, "schwarzschild_ratio": 0.97 }
  ],
  "global_max_density": 5.8,
  "total_persons":      14200
}

Server → Client (alerte critique) :
{
  "type":            "schwarzschild_alert",
  "zone":            "C2",
  "density":         6.3,         // > 6 pers/m² = §5.6 seuil
  "P_TOV_N_m":       5100,        // > seuil physiologique §5.5
  "alert_level":     "ROUGE",
  "message":         "Densité critique franchie — évacuation zone C2 IMMÉDIATE",
  "evacuation_path": { "nearest_exit": "EXIT_3", "time_s": 45 },
  "sms_dispatched":  true
}
```

### 4.3 POST `/simulate/scenario` — Simulation panique

```json
// Requête :
{
  "venue_id":    "stade_olympique",
  "n_persons":   45000,
  "scenario":    "panic_exit",
  "trigger": { "zone": "sector_N", "cause": "alarm" },
  "duration_s":  1800,
  "dt_s":        0.5
}

// Réponse :
{
  "peak_density":           7.2,    // § Schwarzschild §5.6 dépassé
  "peak_time_s":            180,
  "bottleneck_zones":       ["EXIT_2_approach", "corridor_E"],
  "tov_max_pressure_N_m":   5800,   // § §5.5 pression maximale
  "evacuation_time_p90_s":  680,    // 90% évacués en 11 min
  "casualties_risk":        "élevé — intervention requise à t+120s",
  "optimal_exit_redistrib": { "EXIT_1": 0.35, "EXIT_2": 0.25, "EXIT_3": 0.40 },
  "vtk_url":                "s3://crowdsafe/uuid/sim.vtk"
}
```

### 4.4 SLA de performance

| Indicateur | Valeur cible | Méthode de validation |
|---|---|---|
| Densité temps réel | Latence < 100 ms (caméras → alerte) | Test flux vidéo 25 fps |
| Alerte Schwarzschild §5.6 | < 2 s après dépassement seuil | Benchmark pipeline capteurs |
| Simulation 50k agents N-corps | < 30 s / 30 min simulées | Profiling Barnes-Hut 2D |
| Chemin évacuation §4.4 | < 500 ms depuis position quelconque | Test sur 1000 positions aléatoires |
| Précision densité (comptage) | ±10% vs comptage manuel | Validation sur 10 événements réels |

---

## 5. Feuille de route — 5 mois

```
Mois 1     ████████░░░░░░  Phase 1 — N-corps §1 + Schwarzschild §5.6
Mois 2     ░░░░████████░░  Phase 2 — TOV §5.5 + Géodésique §4.4
Mois 3     ░░░░░░░░██████  Phase 3 — Intégration capteurs + temps réel
Mois 4     ░░░░░░░░░░████  Phase 4 — API + Dashboard + alertes
Mois 5     ░░░░░░░░░░░░██  Phase 5 — Pilote stade + open-source
```

### Phase 1 — N-corps §1 et détection densité critique §5.6 (Mois 1)
**Livrables :** Modèle N-corps validé · §5.6 détecte seuils · Émergences spontanées · TRL 5

- `CrowdNBody` §1 : Mesa ABM avec piétons comme particules gravitationnelles
- Barnes-Hut 2D O(N log N) · 50k agents en < 200ms · Cython accéléré
- `CriticalDensity` §5.6 : surveillance temps réel densité ρ_c = 6 pers/m²
- Validation émergences : lane-formation, arching, faster-is-slower
- Comparaison vs modèle de force sociale de Helbing (référence académique)

### Phase 2 — TOV §5.5 et géodésique d'évacuation §4.4 (Mois 2)
**Livrables :** Pression TOV validée · Géodésiques d'évacuation · TRL 6

- `TOVPressure` §5.5 : profil de pression dans les couloirs · seuil 4 450 N/m
- `EvacGeodesic` §4.4 : Dijkstra pondéré par densité · réutilisation code C-02/C-12
- `DensityPotential` §1.3 : carte Φ(x,y) · gradient = direction d'évacuation
- Validation TOV sur données expérimentales BaSiGo (Jülich, DE)
- Validation géodésique sur exercice d'évacuation CNPP (Centre National de Prévention)

### Phase 3 — Intégration capteurs et temps réel (Mois 3)
**Livrables :** Pipeline capteurs → alertes < 2s · Comptage vidéo · TRL 6

- Vision par ordinateur : comptage YOLO v9 depuis flux CCTV (OpenCV)
- Capteurs pression au sol (IoT MQTT) → densité locale
- Pipeline : image → densité → §5.6 → alerte en < 2 s
- Fusion de données multi-sources (caméra + capteurs + billetterie)
- Test sur flux vidéo d'événements archivés (Stade de France, Zénith)

### Phase 4 — API, Dashboard et alertes (Mois 4)
**Livrables :** API production · Dashboard opérateur · WebSocket · TRL 7

- FastAPI REST (6 endpoints) + WebSocket alertes 2 Hz
- Dashboard React : heatmap densité + profil TOV + chemins évacuation
- Intégration plan de salle DXF/GeoJSON/IFC (BIM)
- Alertes push SMS/email aux responsables sécurité
- Mode simulation : what-if scénarios de panique, incidents

### Phase 5 — Pilote stade et open-source (Mois 5)
**Livrables :** Pilote stade · Validation croisée CNPP · Apache 2.0 · Article soumis · TRL 7

- Pilote en conditions réelles : Stade Groupama (OL, Lyon, 59 000 places)
- Validation ±10% vs comptage manuel sur 3 matchs
- Partenariat CNPP (Centre National de Prévention et Protection) · certification
- Publication open-source **Apache 2.0** + article soumis *Safety Science* ou *Fire Safety Journal*

### Jalons critiques

| Semaine | Jalon mesurable |
|---|---|
| S4 | Lane-formation et arching spontanés en simulation N-corps §1 (sans règles codées) |
| S6 | §5.6 alerte rouge ≤ 2s après passage de ρ = 6 pers/m² sur données archivées |
| S8 | TOV §5.5 pression < ±15% vs mesures capteurs BaSiGo Jülich |
| S10 | Géodésique d'évacuation optimale validée vs exercice CNPP |
| S14 | Pipeline caméra → alerte < 2s · ±10% comptage vs manuel (10 événements) |
| S18 | API production + Dashboard · Simulation 50k agents < 30s |
| S20 | Pilote Groupama · ±10% comptage · Open-source publié |

---

## 6. Analyse des risques

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| **Fausse alarme critique** : alerte rouge non justifiée lors d'un match → panique induite | 🔴 Élevée | 🔴 Critique | Double seuil §5.6 + §5.5 requis · Validation humaine obligatoire pour évacuation · Alarme silencieuse d'abord (opérateur) |
| **Latence caméra > 2s** : réseau CCTV hétérogène, RTSP instable | 🟡 Moyenne | 🔴 Élevé | Buffer 3s maximum · Fallback sur capteurs sol si RTSP dropped > 5s · Alarme dégradée |
| **Calibration G_social** : dépend du type d'événement (concert vs sport vs pèlerinage) | 🟡 Moyenne | 🟡 Moyen | G = f(type_event) · base de 5 profils calibrés · Ajustement sur 30 premières minutes |
| **Responsabilité légale** : alerte manquée lors d'une bousculade mortelle | 🟡 Moyenne | 🔴 Critique | Seuils conservateurs · Logs d'audit temps-stamped · Documentation juridique claire |
| **Scalabilité 100k agents** : simulation temps réel insuffisante pour Hajj (2M pers) | 🟢 Faible | 🟡 Moyen | Hiérarchisation : agents réels capteurs + agents simulés macro · GPU CUDA Phase 2 |

### Risque prioritaire : alerte manquée vs fausse alarme

Le dilemme fondamental de CrowdSafe est la tension entre deux erreurs aux conséquences graves :
- **Fausse alarme** : déclenche une évacuation inutile → risque de panique induite
- **Alerte manquée** : ne détecte pas une compression mortelle → blessés/morts

**Stratégie de mitigation à trois niveaux :**

1. **Alerte silencieuse** (ρ > 5 pers/m²) : notification opérateur uniquement
2. **Alerte sonore localisée** (ρ > 6 pers/m² + P_TOV > 3 000 N/m) : annonce dans la zone
3. **Évacuation générale** (ρ > 7 pers/m² pendant > 30s) : décision humaine obligatoire

---

## 7. Budget et indicateurs de succès

### 7.1 Budget estimatif

| Poste | Détail | Montant |
|---|---|---|
| Développement logiciel | 2 développeurs Python/C++ + 1 ingénieur sécurité incendie × 3 mois | 34 000 € |
| Infrastructure cloud | Redis + InfluxDB + PostGIS + GPU simulation (4 mois) | 7 000 € |
| Données et validation | CNPP accès exercices · BaSiGo data · OpenStreetMap venues | 3 000 € |
| Pilote Groupama | Capteurs IoT × 20 · Intégration CCTV · Formation agents | 6 000 € |
| **Total** | | **50 000 €** |

### 7.2 Stack technologique

| Composant | Technologie | Version |
|---|---|---|
| N-corps piétons §1 | Mesa 2.1 + Cython + Barnes-Hut 2D | Python 3.12 |
| TOV / Geodésique | Python 3.12 + SciPy + Dijkstra | SciPy 1.13 |
| Vision par ordinateur | YOLOv9 + OpenCV + RTSP | PyTorch 2.2 |
| Géospatial (salles) | PostGIS + Shapely + DXF/IFC | — |
| Streaming temps réel | Apache Kafka + Redis | — |
| Séries temporelles | InfluxDB | 3.0 |
| Backend API | FastAPI + Uvicorn | Python 3.12 |
| Dashboard | React + Deck.gl + WebGL | — |
| Alertes | Twilio (SMS) + WebSocket | — |

### 7.3 Indicateurs de succès (KPI)

- **Comptage précision** : ±10% vs comptage manuel sur 10 événements archivés
- **Latence alerte** : < 2 s après franchissement ρ_c = 6 pers/m² (§5.6)
- **Fausse alarme** : ≤ 2 alertes rouges injustifiées sur 100 événements simulés
- **Pression TOV** : ±15% vs mesures capteurs BaSiGo (Jülich) sur 5 jeux de données
- **Simulation 50k agents** : < 30 s pour 30 minutes simulées (Barnes-Hut 2D)
- **Évacuation géodésique** : temps prédit dans ±20% du temps réel (exercice CNPP)

---

## Connexions avec la série civile — Réutilisation maximale

CrowdSafe réutilise directement le code de **cinq logiciels précédents** :

| Composant | Source | Logiciel original |
|---|---|---|
| Barnes-Hut N-corps §1 | Code Cython | C-01 CrowdSafe · C-04 EpiDynamic |
| Potentiel Φ §1.3 | Python | C-04 EpiDynamic · C-12 AquaFlow · C-13 UrbanGrow |
| Géodésique RK4 §4.4 | C++/Python | C-02 PathFinder · C-09 SeismoTensor · C-12 AquaFlow |
| TOV ODE Radau §5.5 | SciPy | C-04 EpiDynamic · C-11 AtmoSim · C-12 AquaFlow |
| Mesa ABM | Python | C-01, C-03, C-04, C-11, C-12, C-13 |

Cette réutilisation explique le budget le plus bas de la série pour un logiciel de sécurité critique : **50 k€** vs 55–90 k€ pour les logiciels de même complexité algorithmique.

---

## Références — Corpus Janus

| Référence | Formule | Rôle dans CrowdSafe |
|---|---|---|
| §1.1 | $F = G \cdot mM/d^2$ | Force d'interaction entre piétons |
| §1.2 | Attraction (+,+) · Répulsion (+,−) | Comportement social, évitement obstacles |
| §1.3 | $\Phi^\pm(r) = \mp G|M|/r$ | Carte de pression et risque Φ(x,y) |
| §4.3 | $\Gamma^\sigma_{\mu\nu} = \frac{1}{2}g^{\sigma\rho}(\cdots)$ | Gradient de densité — réfraction du chemin |
| §4.4 | $d^2x^\mu/d\tau^2 + \Gamma^\mu_{\alpha\beta}(\cdots) = 0$ | Chemin d'évacuation optimal |
| §5.5 | $dp/dr = -(\rho c^2+p)(\cdots)$ | Profil de pression de foule dans un couloir |
| §5.6 | $r_s = 2GM/c^2$ | Densité critique ρ_c = 6 pers/m² |

**Sources :**
- Petit, Margnat & Zejli — *ArXiv 2412.04644* (2024)
- Helbing & Molnár (1995) — *Physical Review E* (modèle de force sociale — référence de comparaison)
- Fruin (1987) — *Pedestrian Planning and Design* (niveaux de service piétons)
- BaSiGo (Jülich Research Center) — données expérimentales évacuation
- CNPP — Centre National de Prévention et Protection

---

*Document généré — Mars 2026 · Claude Sonnet 4.6*
*Extension civile du Modèle Cosmologique de Janus — Phase 4 · Dernier logiciel de la série C-01 à C-14*

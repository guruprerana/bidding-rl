# Reward Functions

## Gridworld

### Multi-policy

For policy \(i\),

$$ \text{reward}_i = \text{bid penalty} + \text{window penalty} + \text{distance improvement reward} + \text{target reached reward} + \text{target expiry penalty} $$

Observe that we can provide dense rewards for each local policy with respect to its objective which is not possible in the single-policy setup.

### Single-policy

$$ \text{reward} = \text{distance improvement reward} + \text{target reached reward} + \text{target expiry penalty} $$

Distance improvement is tested in two modes: closest target and target closest to expiry.

## Assault

### Multi-policy

For policy \(i\),

$$ \text{reward}_i = \text{enemy destroy reward} + \text{life loss penalty} + \text{overheat penalty} + \text{fire while hot penalty} + \text{bid penalty} + \text{window penalty} $$

### Single-policy

$$ \text{reward} = \text{enemy destroy reward} + \text{raw score reward} + \text{life loss penalty} + \text{overheat penalty} + \text{fire while hot penalty} $$

## Air Raid

This section lists only the nonzero reward terms used by `experiment_scripts/airraid_bidding_comparison.py`.

### Shared terms

The comparison uses:

- `enemy_destroy_reward = 10.0`
- `enemy_missile_near_hit_penalty = 10.0`
- `enemy_missile_near_hit_y_margin = 35.0`
- `enemy_missile_near_hit_x_radius = 18.0`

For one environment step, a positive score delta is treated as one or more enemy kills:

$$
\text{enemy destroy reward} =
\begin{cases}
\texttt{enemy\_destroy\_reward} \cdot \frac{\Delta \text{score}}{25}, & \Delta \text{score} > 0 \\
0, & \text{otherwise}
\end{cases}
$$

The enemy missile near-hit penalty applies once when an enemy missile enters the near-hit box around the player without causing a life loss:

$$
\text{enemy missile near-hit penalty} =
\begin{cases}
-\texttt{enemy\_missile\_near\_hit\_penalty}, & |x_m - x_p| \le r_x \text{ and } |y_m - y_p| \le m_y \\
0, & \text{otherwise}
\end{cases}
$$

where \(r_x = \texttt{enemy\_missile\_near\_hit\_x\_radius}\) and \(m_y = \texttt{enemy\_missile\_near\_hit\_y\_margin}\).

### Multi-policy

The multi-agent comparison runs `all_pay` and `winner_pays` with:

- `bid_penalty = 0.01`
- `bid_upper_bound = 2`
- `window_penalty = 0.0`

For policy \(i\),

$$
\text{reward}_i =
\text{credited enemy destroy reward}_i
+ \text{controller enemy missile near-hit penalty}_i
+ \text{bid effect}_i
$$

Credit assignment:

- Enemy destroy reward is paid to the lane/agent associated with the hit enemy only when that same agent owns the player missile that caused the score event.
- Enemy missile near-hit penalties are assigned to the current winning/controller agent.
- In `all_pay`, each policy pays `bid_penalty * bid_i` on bidding steps.
- In `winner_pays`, only the winning policy pays `bid_penalty * bid_i` on bidding steps.

### Single-policy

$$
\text{reward} =
\text{enemy destroy reward}
+ \text{enemy missile near-hit penalty}
$$

There are no bid or window penalties in single-agent mode.

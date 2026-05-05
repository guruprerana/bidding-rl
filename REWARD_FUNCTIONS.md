# Reward Functions

## Gridworld

### Multi-policy

For policy \(i\),

$$ \text{reward}_i = \text{bid penalty} + \text{window penalty} + \text{distance improvement reward} + \text{target reached reward} + \text{target expiry penalty} $$

### Single-policy

$$ \text{reward} = \text{distance improvement reward} + \text{target reached reward} + \text{target expiry penalty} $$

Distance improvement is tested in two modes: closest target and target closest to expiry.

## Assault

### Multi-policy

For policy \(i\),

$$ \text{reward}_i = \text{enemy destroy reward} + \text{life loss penalty} + \text{overheat penalty} + \text{fire while hot penalty} + \text{bid penalty} + \text{window penalty} $$

### Single-policy

$$ \text{reward} = \text{enemy destroy reward} + \text{raw score reward} + \text{life loss penalty} + \text{overheat penalty} + \text{fire while hot penalty} $$

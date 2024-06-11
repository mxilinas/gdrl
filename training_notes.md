# Training Notes

## Trial 1
Rewards:
- sync_ep_len: 64
- wall_collision : -1
- agent_collision : 1

No meaningful behavior. Possibly because wall_collision is too strong.

## Trial 2
- sync_ep_len: 64

Agents fell into a pattern of slamming into walls. Probably because there is no 
punishment for touching the walls and the episode len in too small.

## Trial 3
Rewards:
- sync_ep_len: 128
- wall_collision : -0.1
- agent_collision : 1

Agents hugged the floor, should increase wall_collision punishment.

## Trial 4
Rewards:
- sync_ep_len: 128
- wall_collision : -0.5
- agent_collision : 1

## Trail

Shrinking the size of the network is probably a good idea.
Maybe make the wall collision punishment continuous until the agent uncollides 
with the wall. I think a agent collision reward based on force exchange is 
a good idea.

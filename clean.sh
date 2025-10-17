#!/bin/bash
base_dir="/l/users/ahmed.heakl/vit-r1/mcts/checkpoints"

for dir in "$base_dir"/*; do
    if [ -d "$dir" ]; then
        echo "Processing $dir ..."
        cd "$dir" || continue

        # Find all checkpoints (folders starting with checkpoint-)
        checkpoints=($(ls -d checkpoint-* 2>/dev/null | sort -V))
        num_ckpts=${#checkpoints[@]}

        if [ "$num_ckpts" -eq 0 ]; then
            echo "No checkpoints found in $dir"
            continue
        fi

        last_ckpt=${checkpoints[-1]}

        for ckpt in "${checkpoints[@]}"; do
            # Keep last checkpoint and all 5k ones
            if [[ "$ckpt" == "$last_ckpt" || "$ckpt" == *5000 ]]; then
                echo "Keeping $ckpt"
            else
                echo "Deleting $ckpt"
                rm -rf "$ckpt"
            fi
        done
    fi
done

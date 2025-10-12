rsync -a --include='*.md' \
         --include='styles.css' \
         --include='ads.txt' \
         --exclude='*' \
         ./ docs/
        

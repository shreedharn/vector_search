rm -rf site/

mkdocs build

rsync -a --include='*.md' \
         --include='styles.css' \
         --include='ads.txt' \
         --exclude='*' \
         ./ docs/
        

rsync -a --include='*.md' \
         --exclude='*' \
         ./ docs/

rsync -a --include='styles/overrides.css' \
         --exclude='*' \
         ./ docs/
         
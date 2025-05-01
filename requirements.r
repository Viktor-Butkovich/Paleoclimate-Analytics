packages <- c("tidyverse", "olsrr", "jsonlite", "caret", "here", "forecast", "reshape2", "scales", "patchwork")
for (pkg in packages) {
    if (system.file(package = pkg) == "") {
        cat(paste("Installing", pkg))
        install.packages(pkg)
    }
}
cat("All packages installed")

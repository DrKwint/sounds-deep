#!/bin/bash

sphinx-apidoc -o docs -f -T -e sounds_deep
sphinx-build -b dirhtml docs/ docs_build/
#!/bin/bash
set -e

# Generate Cap'n Proto C++ files from schema definitions

# Set up paths
GENERATED_DIR="$DERIVED_FILE_DIR/Generated"

# Determine architecture - fallback to system arch if CURRENT_ARCH is undefined
ARCH="${CURRENT_ARCH}"
if [ "$ARCH" = "undefined_arch" ] || [ -z "$ARCH" ]; then
    ARCH=$(uname -m)
fi

CAPNP_BUILD_DIR="${BUILT_PRODUCTS_DIR}/capnproto-${ARCH}"
CAPNP_BINARY="$CAPNP_BUILD_DIR/bin/capnp"
CAPNPC_CPP_BINARY="$CAPNP_BUILD_DIR/bin/capnpc-c++"

# Create output directory
mkdir -p "$GENERATED_DIR"

# Check if binaries exist
if [ ! -f "$CAPNP_BINARY" ] || [ ! -f "$CAPNPC_CPP_BINARY" ]; then
    echo "error: Cap'n Proto binaries not found at $CAPNP_BUILD_DIR/bin"
    echo "note: Ensure the Cap'n Proto build phase runs before this phase"
    exit 1
fi

# Generate C++ from all .capnp schema files
echo "note: Generating Cap'n Proto C++ files..."
find "$SRCROOT/Messages" -name "*.capnp" -print0 | while IFS= read -r -d '' schema; do
    echo "note: Processing $(basename "$schema")"
    "$CAPNP_BINARY" compile -o"$CAPNPC_CPP_BINARY":"$GENERATED_DIR" \
        --src-prefix="$SRCROOT/Messages" \
        -I "$CAPNP_BUILD_DIR/include" \
        "$schema"
done

echo "note: Cap'n Proto files generated in $GENERATED_DIR"

#!/bin/bash
set -e

# Build Cap'n Proto from source for all target architectures

VENDOR_DIR="$SRCROOT/Vendor"
CAPNP_VERSION="1.1.0"
CAPNP_ARCHIVE="$VENDOR_DIR/capnproto-c++-$CAPNP_VERSION.tar.gz"

# Determine architectures to build
ARCH_LIST="${ARCHS:-${CURRENT_ARCH:-$(uname -m)}}"

echo "note: Building Cap'n Proto for architectures: $ARCH_LIST"

# Build for each architecture
for ARCH in $ARCH_LIST; do
    # Set up architecture-specific paths in derived data
    CAPNP_SRC="$DERIVED_FILE_DIR/capnproto-c++-$CAPNP_VERSION-$ARCH"
    CAPNP_BUILD="$CAPNP_SRC/build"
    CAPNP_BINARY="$CAPNP_BUILD/bin/capnp"

    # Skip if already built
    if [ -f "$CAPNP_BINARY" ]; then
        echo "note: Cap'n Proto already built for $ARCH"
    else
        # Extract source if needed
        if [ ! -d "$CAPNP_SRC" ]; then
            echo "note: Extracting Cap'n Proto source for $ARCH..."
            mkdir -p "$CAPNP_SRC"
            cd "$CAPNP_SRC"
            tar -xzf "$CAPNP_ARCHIVE" --strip-components=1
        fi

        echo "note: Building Cap'n Proto for $ARCH..."
        
        # Create build directory
        mkdir -p "$CAPNP_BUILD"
        cd "$CAPNP_BUILD"
        
        # Clear Xcode architecture settings
        unset NATIVE_ARCH NATIVE_ARCH_ACTUAL
        
        # Configure for specific architecture
        if [ "$ARCH" = "arm64" ]; then
            export CFLAGS="-arch arm64"
            export CXXFLAGS="-arch arm64"
            export LDFLAGS="-arch arm64"
            export CC="clang -arch arm64"
            export CXX="clang++ -arch arm64"
            HOST_FLAG="--host=aarch64-apple-darwin"
            BUILD_FLAG="--build=aarch64-apple-darwin"
        elif [ "$ARCH" = "x86_64" ]; then
            export CFLAGS="-arch x86_64"
            export CXXFLAGS="-arch x86_64"
            export LDFLAGS="-arch x86_64"
            export CC="clang -arch x86_64"
            export CXX="clang++ -arch x86_64"
            HOST_FLAG="--host=x86_64-apple-darwin"
            BUILD_FLAG="--build=x86_64-apple-darwin"
        else
            HOST_FLAG=""
            BUILD_FLAG=""
        fi
        
        # Configure and build
        "$CAPNP_SRC/configure" --prefix="$CAPNP_BUILD" $HOST_FLAG $BUILD_FLAG
        make -j$(sysctl -n hw.ncpu)
        make install
        
        # Clean up environment
        unset CFLAGS CXXFLAGS LDFLAGS CC CXX
    fi
    
    # Copy to build products directory with architecture-specific path
    CAPNP_DEST="${BUILT_PRODUCTS_DIR}/capnproto-$ARCH"
    mkdir -p "$CAPNP_DEST"/{include,lib,bin}
    
    cp -R "$CAPNP_BUILD/include/." "$CAPNP_DEST/include/"
    cp -R "$CAPNP_BUILD/lib/." "$CAPNP_DEST/lib/"
    cp -R "$CAPNP_BUILD/bin/." "$CAPNP_DEST/bin/"
done

echo "note: Cap'n Proto build complete"
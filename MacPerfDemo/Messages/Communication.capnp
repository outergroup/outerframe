@0xcd7293ac51b44617;

struct UUID {
  low @0 :UInt64;
  high @1 :UInt64;
}

struct RangeIDPair {
  location @0 :UInt64;
  length @1 :UInt64;
  id @2 :UUID;
}

struct LocationCanvasPair {
  location @0 :UInt64;
  id @1 :UUID;
  width @2 :UInt32;
  height @3 :UInt32;
  serviceName @4 :Text;  # Bootstrap service name for IOSurface
}

# Message definitions for OuterdocViewer <-> OuterContent communication

# Messages from OuterdocViewer to OuterContent
struct OuterdocMessage {
  union {
    textMessage @0 :Text;
    createDynamicSubstrings @1 :CreateDynamicSubstringsRequest;
    createDynamicSubstring @2 :CreateDynamicSubstringRequest;
    disposeSubstring @3 :DisposeSubstringRequest;
  }
}

# Messages from OuterContent to OuterdocViewer
struct OuterContentMessage {
  union {
    createDynamicSubstringsResponse @0 :CreateDynamicSubstringsResponse;
    createDynamicSubstringResponse @1 :CreateDynamicSubstringResponse;
    updateElements @2 :UpdateElementsMessage;
  }
}

# Request message structures
struct CreateDynamicSubstringsRequest {
  substringIds @0 :List(UUID);
  fileTypes @1 :List(Text);
}

struct CreateDynamicSubstringRequest {
  substringId @0 :UUID;
  fileType @1 :Text;
}

struct DisposeSubstringRequest {
  substringId @0 :UUID;
}

# Response message structures
struct CreateDynamicSubstringsResponse {
  substringIds @0 :List(UUID);
  status @1 :Text;
  results @2 :List(CreateDynamicSubstringResult);
}

struct CreateDynamicSubstringResult {
  attributedString @0 :List(Data);  # Serialized NSAttributedString
  elements @1 :List(RangeIDPair);
  canvases @2 :List(LocationCanvasPair);
}

struct CreateDynamicSubstringResponse {
  substringId @0 :UUID;
  attributedString @1 :Data;  # Serialized NSAttributedString
  elements @2 :List(RangeIDPair);
  canvases @3 :List(LocationCanvasPair);
}

struct UpdateElementsMessage {
  substringId @0 :UUID;
  elementIds @1 :List(UUID);
  newContents @2 :List(Data);  # Serialized NSAttributedString for each element
}

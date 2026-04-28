@0xdf638a93c4a70025;

struct FontProperties {
  family @0 :Text;
  size @1 :Float32;
  weight @2 :UInt32;
  italic @3 :Bool;
}

struct ColorValue {
  red @0 :UInt8;
  green @1 :UInt8;
  blue @2 :UInt8;
  alpha @3 :UInt8;
}

enum Color {
  labelColor @0;
  secondaryLabelColor @1;
  tertiaryLabelColor @2;
  quaternaryLabelColor @3;
  quinaryLabelColor @4;
  systemRedColor @5;
  systemGreenColor @6;
  systemBlueColor @7;
  systemOrangeColor @8;
  systemYellowColor @9;
  systemBrownColor @10;
}

struct Attribute {
  union {
    font @0 :FontProperties;
    foregroundColor @1 :Color;
    backgroundColor @2 :ColorValue;
    underline @3 :Bool;
    strikethrough @4 :Bool;
    superscript @5 :Bool;
    subscript @6 :Bool;
    link @7 :Text;
    paragraph @8 :ParagraphStyle;
    baselineOffset @9 :Float32;
  }
}

struct ParagraphStyle {
  alignment @0 :UInt8;  # 0: left, 1: right, 2: center, 3: justify
  lineSpacing @1 :Float32;
  firstLineHeadIndent @2 :Float32;
  headIndent @3 :Float32;
  tailIndent @4 :Float32;
  lineHeightMultiple @5 :Float32;
  minimumLineHeight @6 :Float32;
  maximumLineHeight @7 :Float32;
}

struct RangeAttributesPair {
  location @0 :UInt64;
  length @1 :UInt64;
  attributes @2 :List(Attribute);
}

struct LibraryEntry {
  id @0 :Text;
  parentDirectoryBookmark @1 :Text;  # Base64-encoded security-scoped bookmark for parent directory
  lastPathComponent @2 :Text;        # The last path component (e.g. project.xcodeproj)
}

struct LibraryRegistry {
  libraries @0 :List(LibraryEntry);
}


struct LocationDataPair {
  location @0 :UInt64;
  data @1 :Data;
}

struct PlatformUrlPair {
  platform @0 :Text;
  url @1 :Text;
}

struct OuterdocMultiAttachment {
  fileType @0 :Text;
  locations @1 :List(LocationDataPair);
}

struct OuterdocAttributedString {
  string @0 :Text;
  attributes @1 :List(RangeAttributesPair);
  attachments @2 :List(OuterdocMultiAttachment);
}

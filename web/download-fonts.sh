#!/bin/bash
# Downloads self-hosted fonts for the bb_rag landing page.
# Run once from the web/ directory: bash download-fonts.sh

set -e
cd "$(dirname "$0")"
mkdir -p fonts

echo "Downloading JetBrains Mono..."
JBMONO_VER="2.304"
JBMONO_URL="https://github.com/JetBrains/JetBrainsMono/releases/download/v${JBMONO_VER}/JetBrainsMono-${JBMONO_VER}.zip"
curl -sL "$JBMONO_URL" -o /tmp/jbmono.zip
unzip -jo /tmp/jbmono.zip "fonts/webfonts/JetBrainsMono-Regular.woff2" -d fonts/
unzip -jo /tmp/jbmono.zip "fonts/webfonts/JetBrainsMono-Light.woff2"   -d fonts/
unzip -jo /tmp/jbmono.zip "fonts/webfonts/JetBrainsMono-Medium.woff2"  -d fonts/
unzip -jo /tmp/jbmono.zip "fonts/webfonts/JetBrainsMono-Bold.woff2"    -d fonts/
rm /tmp/jbmono.zip
echo "  ✓ JetBrains Mono"

echo "Downloading Instrument Sans (variable font, covers 400–600)..."
# Must send a Chrome user-agent — Google Fonts serves woff2 only to modern browsers
UA2="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# Fetch the CSS to get the current URL (Google Fonts rotates paths)
ISANS_URL=$(curl -sL "https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600&display=swap" \
  -H "User-Agent: $UA2" | grep "url(" | grep -v "KpA" | head -1 | sed "s/.*url(\([^)]*\)).*/\1/")
curl -sL "$ISANS_URL" -H "User-Agent: $UA2" -o "fonts/InstrumentSans.woff2"
echo "  ✓ Instrument Sans"

echo ""
echo "Fonts ready in web/fonts/. The landing page will use self-hosted fonts."
ls -lh fonts/*.woff2

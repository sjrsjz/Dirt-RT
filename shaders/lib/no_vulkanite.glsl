uniform float viewWidth;
uniform float viewHeight;
uniform float frameTimeCounter;

const vec3 SUN_DIRECTION = normalize(vec3(1, 3, 2));
const vec3 AMBIENT = vec3(0.1, 0.15, 0.2);

const vec3 LIGHT_POS = vec3(0, 2.0, 0.0);
const vec3 LIGHT_COLOR = vec3(1, 0.8, 0.18) * 25.0;

const vec3 SKY_COLOR = vec3(0.02, 0.05, 0.13);

const float PI = 3.141592654;

const uint _A     = 0x747f18c4u;
const uint _B     = 0xf47d18f8u;
const uint _C     = 0x746108b8u;
const uint _D     = 0xf46318f8u;
const uint _E     = 0xfc39087cu;
const uint _F     = 0xfc390840u;
const uint _G     = 0x7c2718b8u;
const uint _H     = 0x8c7f18c4u;
const uint _I     = 0x71084238u;
const uint _J     = 0x084218b8u;
const uint _K     = 0x8cb928c4u;
const uint _L     = 0x8421087cu;
const uint _M     = 0x8eeb18c4u;
const uint _N     = 0x8e6b38c4u;
const uint _O     = 0x746318b8u;
const uint _P     = 0xf47d0840u;
const uint _Q     = 0x74631934u;
const uint _R     = 0xf47d18c4u;
const uint _S     = 0x7c1c18b8u;
const uint _T     = 0xf9084210u;
const uint _U     = 0x8c6318b8u;
const uint _V     = 0x8c62a510u;
const uint _W     = 0x8c635dc4u;
const uint _X     = 0x8a88a8c4u;
const uint _Y     = 0x8a884210u;
const uint _Z     = 0xf844447cu;
const uint _a     = 0x0382f8bcu;
const uint _b     = 0x85b318f8u;
const uint _c     = 0x03a308b8u;
const uint _d     = 0x0b6718bcu;
const uint _e     = 0x03a3f83cu;
const uint _f     = 0x323c8420u;
const uint _g     = 0x03e2f0f8u;
const uint _h     = 0x842d98c4u;
const uint _i     = 0x40308418u;
const uint _j     = 0x080218b8u;
const uint _k     = 0x4254c524u;
const uint _l     = 0x6108420cu;
const uint _m     = 0x06ab5ac4u;
const uint _n     = 0x07a318c4u;
const uint _o     = 0x03a318b8u;
const uint _p     = 0x05b31f40u;
const uint _q     = 0x03671784u;
const uint _r     = 0x05b30840u;
const uint _s     = 0x03e0e0f8u;
const uint _t     = 0x211c420cu;
const uint _u     = 0x046318bcu;
const uint _v     = 0x04631510u;
const uint _w     = 0x04635abcu;
const uint _x     = 0x04544544u;
const uint _y     = 0x0462f0f8u;
const uint _z     = 0x07c4447cu;
const uint _0     = 0x746b58b8u;
const uint _1     = 0x23084238u;
const uint _2     = 0x744c88fcu;
const uint _3     = 0x744c18b8u;
const uint _4     = 0x19531f84u;
const uint _5     = 0xfc3c18b8u;
const uint _6     = 0x3221e8b8u;
const uint _7     = 0xfc422210u;
const uint _8     = 0x745d18b8u;
const uint _9     = 0x745e1130u;
const uint _space = 0x0000000u;
const uint _dot   = 0x000010u;
const uint _minus = 0x0000e000u;
const uint _comma = 0x00000220u;
const uint _colon = 0x02000020u;
const uint _excl  = 0x42108020u;

const int charWidth   = 5;
const int charHeight  = 6;
const int charSpacing = 1;
const int lineSpacing = 1;

const ivec2 charSize  = ivec2(charWidth, charHeight);
const ivec2 spaceSize = charSize + ivec2(charSpacing, lineSpacing);

// Text renderer

struct Text {
	vec4 result;     // Output color from the text renderer
	vec4 fgCol;      // Text foreground color
	vec4 bgCol;      // Text background color
	ivec2 fragPos;   // The position of the fragment (can be scaled to adjust the size of the text)
	ivec2 textPos;   // The position of the top-left corner of the text
	ivec2 charPos;   // The position of the next character in the text
	int base;        // Number base
	int fpPrecision; // Number of decimal places to print
} text;

// Fills the global text object with default values
void beginText(ivec2 fragPos, ivec2 textPos) {
	text.result      = vec4(0.0);
	text.fgCol       = vec4(0.27, 0.88, 1.00, 1.0);
	text.bgCol       = vec4(0.0);
	text.fragPos     = fragPos;
	text.textPos     = textPos;
	text.charPos     = ivec2(0);
	text.base        = 10;
	text.fpPrecision = 2;
}

// Applies the rendered text to the fragment
void endText(inout vec3 fragColor) {
	fragColor = mix(fragColor.rgb, text.result.rgb, text.result.a);
}

void printChar(uint character) {
	ivec2 pos = text.fragPos - text.textPos - spaceSize * text.charPos * ivec2(1, -1) + ivec2(0, spaceSize.y);

	uint index = uint(charWidth - pos.x + pos.y * charWidth + 1);

	// Draw background
	if (clamp(pos, ivec2(0), spaceSize - 1) == pos)
		text.result = mix(text.result, text.bgCol, text.bgCol.a);

	// Draw character
	if (clamp(pos, ivec2(0), charSize - 1) == pos)
		text.result = mix(text.result, text.fgCol, text.fgCol.a * float(character >> index & 1u));

	// Advance to next character
	text.charPos.x++;
}

#define printString(string) {                                               \
	uint[] characters = uint[] string;                                      \
	for (int i = 0; i < characters.length(); ++i) printChar(characters[i]); \
}

float hash13(vec3 p3) {
	p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec3 p) {
    vec3 iPos = floor(p);
    vec3 fPos = smoothstep(0.0, 1.0, fract(p));
    return mix(
        mix(
            mix(
                hash13(iPos + vec3(0, 0, 0)),
                hash13(iPos + vec3(0, 0, 1)),
                fPos.z
            ),
            mix(
                hash13(iPos + vec3(0, 1, 0)),
                hash13(iPos + vec3(0, 1, 1)),
                fPos.z
            ),
            fPos.y
        ),
        mix(
            mix(
                hash13(iPos + vec3(1, 0, 0)),
                hash13(iPos + vec3(1, 0, 1)),
                fPos.z        
            ),
            mix(
                hash13(iPos + vec3(1, 1, 0)),
                hash13(iPos + vec3(1, 1, 1)),
                fPos.z        
            ),
            fPos.y
        ),
        fPos.x
    );
}

float sdCone(vec3 p, vec2 c, float h) {
    vec2 q = h*vec2(c.x/c.y,-1.0);
    
    vec2 w = vec2( length(p.xz), p.y );
    vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
    vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
    float k = sign( q.y );
    float d = min(dot( a, a ),dot(b, b));
    float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
    return sqrt(d)*sign(s);
}

float sdCylinder(vec3 p, vec3 c) {
    return length(p.xz - c.xy) - c.z;
}

float sdPlane(vec3 p, vec3 n, float h) {
    return dot(p,n) + h;
}

float sdCappedCylinder(vec3 p, float h, float r) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float opSmoothSubtraction(float d1, float d2, float k) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h); 
}

float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h); 
}

float sdVolcano(vec3 p) {
    const float angle = radians(40.0);
    const vec2 sincos = vec2(sin(angle), cos(angle));
    float base = sdCone(p - vec3(0, 1.0, 0), sincos, 3.0) - sin(atan(p.z, p.x) * 10.0) * 0.05;
    float hole = sdCylinder(p, vec3(0.0, 0.0, 0.4));
    float cap = sdPlane(p, vec3(0, -1, 0), 0.3);
    return opSmoothSubtraction(
        hole,
        opSmoothSubtraction(
            cap,
            base, 
            0.1
        ), 
        0.1
    );
}

float sdGround(vec3 p) {
    return sdCappedCylinder(p - vec3(0, -1.2, 0), 0.1, 3.0);
}


struct DF {
    float dist;
    vec3 albedo;
    bool lit;
};

DF sdLava(vec3 p) {
    const vec3 bottomColor = vec3(1, 0.8, 0.18);
    const vec3 topColor = vec3(1, 0.42, 0.15);
    
    const float angle = radians(30.0);
    const vec2 sincos = vec2(sin(angle), cos(angle));
    
    float noiseVal = sin(atan(p.z, p.x) * 10.0);
    
    return DF(
        opSmoothUnion(
            sdCappedCylinder(p, 2.0, 0.3),
            sdCone(p * vec3(1, -1, 1) + vec3(0, 1, 0), sincos, 1.5) - noiseVal * 0.02,
            0.3
        ),
        mix(bottomColor, topColor, smoothstep(0.5, 2.5, p.y + noiseVal * 0.2 + noise(p * 10.0) * 0.5 - 0.25)),
        false
    );
}

DF map(vec3 p) {
    float volcano = sdVolcano(p);
    float ground = sdGround(p);
    DF lava = sdLava(p);
    if (volcano < ground && volcano < lava.dist) {
        return DF(volcano, vec3(0.35, 0.11, 0) - noise(p * 10.0 + 13.0) * 0.1, true);
    } else if (ground < lava.dist) {
        return DF(ground, vec3(0.07, 0.48, 0.04) - noise(p * 5.0) * 0.1, true);
    } else {
        return lava;
    }
}

vec3 getNormal(vec3 p) {
    vec2 e = vec2(0.01, 0.0);
    return normalize(vec3(
        map(p + e.xyy).dist - map(p - e.xyy).dist,
        map(p + e.yxy).dist - map(p - e.yxy).dist,
        map(p + e.yyx).dist - map(p - e.yyx).dist
    ));
}

mat3 lookAt(vec3 eye, vec3 target) {
    vec3 forward = normalize(eye - target);
    vec3 right = cross(vec3(0, 1, 0), forward);
    vec3 up = cross(forward, right);
    return mat3(right, up, forward);
}

vec2 cylIntersect(in vec3 ro, in vec3 rd, in vec3 cb, in vec3 ca, float cr) {
    vec3  oc = ro - cb;
    float card = dot(ca,rd);
    float caoc = dot(ca,oc);
    float a = 1.0 - card*card;
    float b = dot( oc, rd) - caoc*card;
    float c = dot( oc, oc) - caoc*caoc - cr*cr;
    float h = b*b - a*c;
    if( h<0.0 ) return vec2(-1.0);
    h = sqrt(h);
    return vec2(-b-h, -b+h) / a;
}

struct TextPoint {
    float dist;
    vec3 color;
};

TextPoint getText(float dist, vec3 point) {
    if (clamp(point.y, 0.1, 0.3) != point.y)
        return TextPoint(-1.0, vec3(0.0));
    vec2 uv = vec2(
        atan(point.z, point.x),
        (point.y - 0.1) / 0.2
    );
    uv.x = mod(1.0 - uv.x, PI);
    
    if (mod(frameTimeCounter, 3.0) > 2.5) {
        uv += (noise(floor(point + vec3(frameTimeCounter * 5.0)) * 50.0) * 0.8 - 0.4) * vec2(0.1, 1.0);
    }
    
    beginText(ivec2(uv * vec2(70.0, 6.0)), ivec2(0, 7));
    printString((_W, _a, _r, _n, _i, _n, _g, _excl, _space, _T, _h, _i, _s, _space, _p, _a, _c, _k, _space, _n, _e, _e, _d, _s, _space, _V, _u, _l, _k, _a, _n, _i, _t, _e, _excl));
    vec3 color = vec3(1, 0, 1);
    endText(color);
    if (dot(color - vec3(1, 0, 1), color - vec3(1, 0, 1)) < 0.01) {
        return TextPoint(-1.0, vec3(0.0));
    }
    
    return TextPoint(dist, color);
}

vec3 raymarch(vec3 origin, vec3 direction) {
    vec3 p = origin;
    
    vec2 textDist = cylIntersect(origin, direction, vec3(0.0), vec3(0, 1, 0), 1.5);
    vec3 hp1 = origin + textDist.x * direction;
    vec3 hp2 = origin + textDist.y * direction;
    TextPoint tp1 = getText(textDist.x, hp1);
    TextPoint tp2 = getText(textDist.y, hp2);
    
    float totalDist = 0.0;
    for (int i = 0; i < 35; i++) {
        bool hitText = false;
        vec3 hitPoint;
        if (tp1.dist > 0.0 && totalDist > tp1.dist) {
            return tp1.color;
        }
        if (tp2.dist > 0.0 && totalDist > tp2.dist) {
            return tp2.color;
        }
    
        DF df = map(p);
        
        if (df.dist < 0.03) {
            if (!df.lit)
                return df.albedo;
            vec3 normal = getNormal(p);
            vec3 toLight = LIGHT_POS - p;
            float lenSqr = dot(toLight, toLight);
            toLight /= sqrt(lenSqr);
            float diffuse = max(dot(normal, toLight), 0.0);
            return (AMBIENT + diffuse * LIGHT_COLOR / lenSqr) * df.albedo;
        }
        const float stepFrac = 0.9;
        totalDist += df.dist * stepFrac;
        p += direction * df.dist * stepFrac;
    }
   
    return mix(SKY_COLOR, vec3(1.0), step(0.97, noise(direction * 60.0)));
}

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

void main() {
    vec2 resolution = vec2(viewWidth, viewHeight);
    vec3 direction = normalize(vec3(
        (gl_FragCoord.xy - resolution / 2.0) / resolution.y,
        -1.0
    ));
    
    float angle = -frameTimeCounter * 0.2;
    vec3 origin = vec3(
        cos(angle) * 6.0,
        2.0,
        sin(angle) * 6.0
    );
    
    direction = lookAt(origin, vec3(0.0)) * direction;
    fragColor.rgb = raymarch(origin, direction);
    fragColor.a = 1.0;
}
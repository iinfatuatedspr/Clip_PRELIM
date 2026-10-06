"""Extract metadata/val/test from CODEBRIM zip by scanning for local file headers
(the central directory's offsets are unreliable in this archive). Every file is CRC-checked."""
import mmap, os, struct, zipfile, zlib

PATH, OUT = 'downloads/codebrim.zip', 'data/codebrim'
KEEP = {'metadata', 'val', 'test'}
CH = 1 << 20

def wanted(n):
    parts = n.split('/')
    return (not n.startswith('__MACOSX') and not n.endswith('/')
            and parts[-1] != '.DS_Store' and any(p in KEEP for p in parts[:-1]))

z = zipfile.ZipFile(PATH)
want = {i.orig_filename: i for i in z.infolist() if wanted(i.filename)}
print('files wanted:', len(want))

f = open(PATH, 'rb')
m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
found, pos, hits = {}, 0, 0
while True:
    pos = m.find(b'PK\x03\x04', pos)
    if pos < 0:
        break
    hits += 1
    name_len, extra_len = struct.unpack('<HH', m[pos + 26:pos + 30])
    name = m[pos + 30:pos + 30 + name_len].decode('ascii', 'replace')
    if name in want and name not in found:
        found[name] = (pos, name_len, extra_len)
    pos += 4
print(f'local headers scanned: {hits}, wanted files located: {len(found)} / {len(want)}')
missing = [n for n in want if n not in found]
if missing:
    print('MISSING', len(missing), 'e.g.', missing[:5])

ok = bad = 0
for k, (name, (pos, nl, el)) in enumerate(found.items()):
    info = want[name]
    method = struct.unpack('<H', m[pos + 8:pos + 10])[0]
    start = pos + 30 + nl + el
    if method == 0:
        data = m[start:start + info.file_size]
    elif method == 8:
        d, out, p = zlib.decompressobj(-15), [], start
        while not d.eof and p < len(m):
            out.append(d.decompress(m[p:p + CH]))
            p += CH
        data = b''.join(out)
    else:
        print('unsupported compression', method, name); bad += 1; continue
    if zlib.crc32(data) != info.CRC or len(data) != info.file_size:
        print('CRC/size mismatch:', name); bad += 1; continue
    dst = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as g:
        g.write(data)
    ok += 1
    if k % 200 == 0:
        print(k, name)
print(f'done: {ok} ok, {bad} bad, {len(missing)} missing')

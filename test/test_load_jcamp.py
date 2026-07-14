import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperlyse.database import Database, Metadata, Spectrum
from hyperlyse.feature_extractor import FeatureExtractor
from hyperlyse.vector_store import VectorStore

HERE = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Standard '(X++(Y..Y))' AFFN format (e.g. exported by OPUS)
# ---------------------------------------------------------------------------

FOREIGN_JDX = """\
##JCAMP-DX=4.24
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##XFACTOR=1
##YFACTOR=2
##DELTAX=1
##FIRSTX=10
##LASTX=15
##NPOINTS=6
##XYDATA=(X++(Y..Y))
10+1+2+3
13+4+5+6
##END=
"""


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def test_foreign_xppyy_format(tmp_path):
    """One X + many '+'-delimited Y per line, headers without a space after '='."""
    f = _write(tmp_path, 'opus_export.dx', FOREIGN_JDX)
    s = Spectrum.load_jcamp(f)
    # X increments by DELTAX across the Y values on each line
    np.testing.assert_array_equal(s.x, [10, 11, 12, 13, 14, 15])
    # Y is scaled by YFACTOR (=2)
    np.testing.assert_array_equal(s.y, [2, 4, 6, 8, 10, 12])


def test_foreign_npoints_matches_header(tmp_path):
    f = _write(tmp_path, 'opus_export.dx', FOREIGN_JDX)
    s = Spectrum.load_jcamp(f)
    assert len(s.x) == 6 and len(s.y) == 6


def test_deltax_derived_when_missing(tmp_path):
    """When ##DELTAX is absent, it is derived from FIRSTX/LASTX/NPOINTS."""
    content = FOREIGN_JDX.replace('##DELTAX=1\n', '')
    f = _write(tmp_path, 'opus_nodelta.dx', content)
    s = Spectrum.load_jcamp(f)
    np.testing.assert_array_equal(s.x, [10, 11, 12, 13, 14, 15])


def test_id_falls_back_to_filename(tmp_path):
    """A foreign file without ##TITLE gets its id from the filename."""
    f = _write(tmp_path, 'cochineal_lake.dx', FOREIGN_JDX)
    s = Spectrum.load_jcamp(f)
    assert s.metadata.id == 'cochineal_lake'


# ---------------------------------------------------------------------------
# Backward compatibility: Hyperlyse's own 'X Y'-pair output
# ---------------------------------------------------------------------------

def test_hyperlyse_pair_format_roundtrip(tmp_path):
    """A spectrum saved by Hyperlyse must load back with identical values."""
    # save_jcamp stores values as float32; use exactly-representable values so
    # the save -> str -> load round-trip is bit-exact.
    x = np.arange(400, 460, dtype=np.float32)          # integer wavelengths
    y = (np.arange(60) / 256.0).astype(np.float32)     # k / 2^n are exact
    src = Spectrum(x, y, Metadata('sample_1', source_object='obj'))
    f = str(tmp_path / 'roundtrip.jdx')
    src.save_jcamp(f)
    loaded = Spectrum.load_jcamp(f)
    np.testing.assert_array_equal(loaded.x, x.astype(np.float64))
    np.testing.assert_array_equal(loaded.y, y.astype(np.float64))
    assert loaded.metadata.id == 'sample_1'
    assert loaded.metadata.source_object == 'obj'


@pytest.mark.parametrize('fixture', [
    'bull_bottomrightpixel_hyperlyse_newx.jdx',
    'bull_bottomrightpixel_hyperlyse_newx_nodelta.jdx',
])
def test_existing_hyperlyse_fixtures(fixture):
    """Committed Hyperlyse-format fixtures parse to the same X/Y as a naive
    pair parse (guards the backward-compatible path)."""
    path = os.path.join(HERE, fixture)
    s = Spectrum.load_jcamp(path)
    ex, ey, started = [], [], False
    for line in open(path):
        line = line.rstrip('\n')
        if line.startswith('##XYDATA'):
            started = True
            continue
        if line.startswith('##END'):
            break
        if started and not line.startswith('##'):
            parts = line.split()
            if len(parts) == 2:
                ex.append(float(parts[0]))
                ey.append(float(parts[1]))
    np.testing.assert_array_equal(s.x, np.array(ex))
    np.testing.assert_array_equal(s.y, np.array(ey))


# ---------------------------------------------------------------------------
# Robustness: empty / malformed spectra must not crash search
# ---------------------------------------------------------------------------

def _make_db(spectra, tmp_path):
    db = Database.__new__(Database)
    db.root = ''
    db._extractor = FeatureExtractor()
    db._store = VectorStore(str(tmp_path / 'cache'))
    db.spectra = spectra
    return db


def test_search_skips_empty_spectrum(tmp_path):
    """An empty spectrum in the DB is skipped instead of raising IndexError."""
    np.random.seed(0)
    good = Spectrum(np.linspace(400, 900, 200), np.random.rand(200), Metadata('good'))
    empty = Spectrum(np.array([]), np.array([]), Metadata('empty'))
    db = _make_db([empty, good], tmp_path)
    results = db.search_spectrum(np.linspace(420, 880, 180), np.random.rand(180))
    ids = [r['spectrum'].metadata.id for r in results]
    assert 'good' in ids
    assert 'empty' not in ids


def test_search_all_empty_returns_no_results(tmp_path):
    empty = Spectrum(np.array([]), np.array([]), Metadata('empty'))
    db = _make_db([empty], tmp_path)
    results = db.search_spectrum(np.linspace(420, 880, 180), np.random.rand(180))
    assert results == []


def test_compare_spectra_empty_returns_none():
    x = np.linspace(400, 800, 50)
    y = np.random.rand(50)
    assert Database.compare_spectra(x, y, np.array([]), np.array([])) is None
    assert Database.compare_spectra(np.array([]), np.array([]), x, y) is None

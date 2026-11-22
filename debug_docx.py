import sys
sys.path.insert(0, r'E:/projects/airag/src')
from airag.ingestion.loaders.docx import DocxLoader
import docx
import tempfile
from pathlib import Path
import zipfile
import re

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    path = Path(tmp) / 'no_core.docx'
    stripped = Path(tmp) / 'stripped.docx'
    document = docx.Document()
    document.add_paragraph('Body')
    document.save(path)

    with zipfile.ZipFile(path) as source, zipfile.ZipFile(
        stripped, 'w', zipfile.ZIP_DEFLATED
    ) as target:
        for item in source.infolist():
            if item.filename == 'docProps/core.xml':
                continue
            data = source.read(item.filename)
            if item.filename == '_rels/.rels':
                rels = data.decode('utf-8')
                rels = re.sub(
                    r'<Relationship[^>]*Type="[^"]*/core-properties"[^>]*/>',
                    '',
                    rels,
                )
                data = rels.encode('utf-8')
            target.writestr(item, data)

    loader = DocxLoader()
    doc = docx.Document(str(stripped))
    core = doc.core_properties
    print('core:', core)
    print('title:', core.title)
    print('author:', core.author)
    print('created:', core.created, type(core.created))
    print('modified:', core.modified, type(core.modified))

    from airag.ingestion.loaders.docx import DocxLoader
    loader = DocxLoader()
    title = loader._core_prop(doc.core_properties, 'title')
    author = loader._core_prop(doc.core_properties, 'author')
    created = loader._core_prop(doc.core_properties, 'created')
    modified = loader._core_prop(doc.core_properties, 'modified')
    print('title:', title, type(title))
    print('author:', author, type(author))
    print('created:', created, type(created))
    print('modified:', modified, type(modified))
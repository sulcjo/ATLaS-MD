# Collective variables

`--cv1` is primary coordinate: `distance`, `contacts`, or `nonlocal-contacts`. Its units are Å for distance and dimensionless for normalized contacts. `--cv2` adds secondary 2D coordinate: `none`, secondary-structure modes, `rama-map`, `custom`, `torsion-pca`, or `tica-linear`.

```bash
gareus --seq GYDPETGTWG --cv1 contacts --cv2 rama-map --out contacts_rama
```

For contact CVs, choose atom selection/scheme deliberately:

```bash
--contact-scheme residue-balanced --contact-atom-selection backbone-heavy \
--contact-min-sequence-separation 3 --contact-r0-a 4.8 --contact-beta-a-inv 2.5
```

`torsion-pca` is bootstrap secondary coordinate from seed conformers. `tica-linear` needs readable learned torsion state; resume must restore exact state. Use adaptive tICA only after reading [adaptive workflow](adaptive-workflows.md).

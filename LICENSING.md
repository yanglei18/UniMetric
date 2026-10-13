# Licensing status before a public release

No root `LICENSE` has been applied yet. The copyright/authorization holder for
the first-party Metric-Depth code has not been confirmed. Publishing the source
on GitHub does **not** by itself grant third parties an open-source license to
copy, modify, or redistribute it. Confirm the author(s) and any institutional
or collaborator rights before adding a project license.

If the authorized holder wants a permissive license, Apache-2.0 is a reasonable
candidate for **first-party code only** because it includes an explicit patent
grant and is OSI-approved. Do not apply it to model source, weights, datasets,
or code for which the project does not hold the required rights. This is a
publication checklist, not a claim that permission has already been granted.

The local `vendor/` source cache is excluded from Git by default. Individual
upstreams have different terms: 4RC/STream3R restrict use to non-commercial
purposes; Scal3R has a project-registration requirement; VGGT/DVLT include
custom or mixed terms; Pi3 code and its weights have different licenses. See
[`vendor/README.md`](vendor/README.md) and each upstream license/model card.
Do not describe the complete nine-model dependency stack as Apache-licensed.

Release checklist:

1. Confirm who can authorize licensing of the original project code and any
   contributed or adapted files.
2. Select and add the approved root license and any required copyright/NOTICE
   text; keep third-party code excluded or retain each required upstream notice.
3. Review dataset redistribution terms and every checkpoint's model card.
4. Re-run the secret scan and inspect the exact `git add`/diff before making
   the GitHub repository public.

References: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0),
[Open Source Definition](https://opensource.org/osd),
[no-license explanation](https://choosealicense.com/no-permission/).

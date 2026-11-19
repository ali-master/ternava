# Changelog

## [0.3.0](https://github.com/ali-master/ternava/compare/v0.2.0...v0.3.0) (2026-11-19)

### Features

* **agents:** add huggingface skills ([2100bc3](https://github.com/ali-master/ternava/commit/2100bc33665561dc6feb5b114b397fd4a69c722c))
* **agents:** add huggingface skills ([2f769b9](https://github.com/ali-master/ternava/commit/2f769b9b52eb738aaf251151c2b6d68fedc3255d))
* **eval:** add tool-search benchmark and checkpoint scoring ([cd94b3c](https://github.com/ali-master/ternava/commit/cd94b3cedf5bad4f6f957b6eb8f4bb441f9f6025))
* **training:** add v2 pilot config and cache-free domain round ([cd7d1a0](https://github.com/ali-master/ternava/commit/cd7d1a0a7473db9fcfce6a1f826c10012ae3c207))

### Bug Fixes

* **docs-search:** load the published bilingual model ([f09d670](https://github.com/ali-master/ternava/commit/f09d670f7500731729cf4f5d06c986e7ee854a92))
* **emoji-search:** rebuild vectors with 0.2.0 ([dd634a5](https://github.com/ali-master/ternava/commit/dd634a50fced1fdff673c596b55baf7d5b3d08d1))
* **release:** wait for the published tarball before syncing the examples ([f6fc610](https://github.com/ali-master/ternava/commit/f6fc6107c641d1434db06c6b66324c3029d33582))
* **tests:** unblock the branding check ([40e5038](https://github.com/ali-master/ternava/commit/40e5038cc3b041245b3867ba71dfa6d3ac914a4c))
* **tokenizer:** map Persian and Arabic-Indic digits to ASCII ([fe58b3b](https://github.com/ali-master/ternava/commit/fe58b3bec33f39bece0c2692e7e0cd989e8c30fb))

## [0.2.0](https://github.com/ali-master/ternava/compare/v0.1.1...v0.2.0) (2026-10-13)

### Features

* **docs-search:** add Persian Iran history corpus ([72156ec](https://github.com/ali-master/ternava/commit/72156ec827055970376917d91650fefac410e2bb))
* **docs-search:** localize history search interface ([aa7091d](https://github.com/ali-master/ternava/commit/aa7091ddc7065e9e1f659a9d4a46be5a971c8427))
* **emoji-search:** add Persian phrases and matching ([84624e2](https://github.com/ali-master/ternava/commit/84624e24e2555d1c23405d43e918073cdef182c3))
* **emoji-search:** localize interface and examples ([77e3929](https://github.com/ali-master/ternava/commit/77e3929b8326d1c35ab67d591e3f4142b25beb0a))
* **engine:** load bilingual model artifacts ([6ba3967](https://github.com/ali-master/ternava/commit/6ba3967020c6dc6fab60c30b9d961b65d259d1f8))
* **eval:** audit held-out data and correlations ([13c3f66](https://github.com/ali-master/ternava/commit/13c3f668bc3dd520b3a60cce6fc3c220d9889621))
* **eval:** evaluate bilingual model quality ([f313018](https://github.com/ali-master/ternava/commit/f3130182af244faa4ac4ba953f4a2ca09f9c0906))
* **eval:** preview Persian models without aliases ([1080d2c](https://github.com/ali-master/ternava/commit/1080d2c9efc4117aef4212cf2f054354c9604c8c))
* **odyssey:** search Persian narrative excerpts ([7ff0d80](https://github.com/ali-master/ternava/commit/7ff0d80eccf8bce237a7c9b2d649ed809b1f9f8e))
* **pack:** support bilingual models and positions ([4f3d322](https://github.com/ali-master/ternava/commit/4f3d3221464513f8f2f8bd185d53891bf99ae60f))
* **runtime:** support Node Bun and browsers ([29f537e](https://github.com/ali-master/ternava/commit/29f537ec3939375361cc98ba8aae80d4903fb33c))
* **training:** add bilingual tokenizer identity ([89572db](https://github.com/ali-master/ternava/commit/89572dbdaaa581c08e6f133f418215e89e474ccc))
* **training:** add compositional contrast training ([b406e03](https://github.com/ali-master/ternava/commit/b406e0398f959d8489df459fca43f95d1f8dddf2))
* **training:** add native Persian continuation ([49ab0e4](https://github.com/ali-master/ternava/commit/49ab0e43f36e7e86bd7ffe6ee46affcdce185fb8))
* **training:** add negation and numeric pilot ([d6d4bf2](https://github.com/ali-master/ternava/commit/d6d4bf2cb7b44d47f05f9b94b7270f141fed8248))
* **training:** add round 2 pilot ([6cd3d32](https://github.com/ali-master/ternava/commit/6cd3d32ca2d015b45c3142d49a0ba2a57b5cdab0))
* **training:** add round 3 pilot ([63124cd](https://github.com/ali-master/ternava/commit/63124cdcb5a5d3b28810f458cbea31b0062a9938))
* **training:** add round 4 pilot ([dd1fc27](https://github.com/ali-master/ternava/commit/dd1fc2766aac2bada412721d3c14a4420b9376f7))
* **training:** add technical Persian distillation ([7066bd5](https://github.com/ali-master/ternava/commit/7066bd51347e764f827b6dfbf2c0d242c2029168))
* **training:** configure bilingual mini distillation ([9c1dc83](https://github.com/ali-master/ternava/commit/9c1dc83d7a41c5b5f4cf75e6d412bd589a36276d))
* **training:** prepare bilingual teacher data ([2e762a0](https://github.com/ali-master/ternava/commit/2e762a04b3e7bc05339d65e851f8f18c44513a4b))
* **training:** preserve tokenizer provenance in runs ([cd77286](https://github.com/ali-master/ternava/commit/cd77286b6b90b0f2100ec59ff7fd9eee439dfcab))

### Bug Fixes

* correct comments and improve readability in index.mjs and search.mjs ([151cd1b](https://github.com/ali-master/ternava/commit/151cd1bfcaadf130864d356d02ff130539250b69))
* **docs-search:** correct RTL search layout ([bcd6af2](https://github.com/ali-master/ternava/commit/bcd6af2df1db3d9367bdea3c101de9234ddcd61f))
* **engine:** match PyTorch activation rounding ([adacc66](https://github.com/ali-master/ternava/commit/adacc665210064791c8ecd23c7d71ab739e8cd53))
* **release:** prevent duplicate version tags ([ff45038](https://github.com/ali-master/ternava/commit/ff450382fe7732a7294a234202f5f00a5169e0ad))
* update demo links ([23da804](https://github.com/ali-master/ternava/commit/23da804d0ae19298c78446619905d92176a4ac83))

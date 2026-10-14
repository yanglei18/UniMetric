"""Installer behavior without Conda, network access, model weights, or GPUs."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load_script('install_gfm')
with mock.patch.dict(sys.modules, {'install_gfm': installer}):
    downloader = load_script('download_gfm_weights')


def arguments(**changes):
    values = dict(env_name=None, dry_run=False, no_weights=False,
                  skip_cuda_check=False, load_model=False, reuse_env=False)
    values.update(changes)
    return SimpleNamespace(**values)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.profiles = installer.load_profiles()

    def test_no_arguments_install_all_models_without_prompting(self):
        for isatty in (False, True):
            with self.subTest(isatty=isatty), mock.patch.object(
                sys, 'argv', ['install_gfm.py']
            ), mock.patch.object(sys.stdin, 'isatty', return_value=isatty), mock.patch(
                'builtins.input', side_effect=AssertionError('must not prompt')
            ), mock.patch.object(installer, 'install_group') as install_group, contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(installer.main(), 0)
            self.assertEqual([call.args[0] for call in install_group.call_args_list], [
                installer.select_models(['common'], self.profiles), ['dvlt'],
            ])
            for call in install_group.call_args_list:
                args = call.args[2]
                self.assertIsNone(args.env_name)
                self.assertFalse(args.no_weights)
                self.assertFalse(args.skip_cuda_check)
            self.assertIn('All nine GFMs installed.', output.getvalue())
            self.assertTrue(output.getvalue().rstrip().endswith('python get_metric_scale.py'))

    def test_all_nine_config_aliases_and_dry_runs_have_no_side_effects(self):
        expected = {
            'pi3_long': 'pi3', 'vggt_long': 'vggt', 'vggt_omega_long': 'omega',
            'da3_long': 'da3', 'scal3r_long': 'scal3r', 'lingbot_map_long': 'lingbot',
            '4rc_long': '4rc', 'stream3r_long': 'stream3r', 'dvlt_long': 'dvlt',
        }
        with contextlib.ExitStack() as stack:
            # Fail if a supposedly read-only plan tries to write or run commands.
            for target in ('subprocess.run', 'subprocess.check_output',
                           'pathlib.Path.mkdir', 'pathlib.Path.write_text',
                           'pathlib.Path.write_bytes', 'pathlib.Path.rename'):
                stack.enter_context(mock.patch(target, side_effect=AssertionError(target)))
            for alias, model in expected.items():
                with self.subTest(alias=alias), contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(installer.select_model(alias, self.profiles), model)
                    for extra_alias in self.profiles['models'][model].get('aliases', []):
                        self.assertEqual(installer.select_model(extra_alias, self.profiles), model)
                    installer.install(model, self.profiles, arguments(dry_run=True))
                    plan = json.loads(output.getvalue())
                    self.assertEqual(plan['models'], [model])
                    self.assertEqual(plan['environment'], 'unimetric-dvlt' if model == 'dvlt' else 'unimetric')
                    self.assertEqual(len(plan['sources']), 2)

    def test_dvlt_overrides_replace_seven_conflicting_base_pins(self):
        requirements = installer.merged_requirements(self.profiles['models']['dvlt'])
        pins = {}
        for requirement in requirements:
            name, version = requirement.split('==')
            name = installer.canonical_name(name)
            self.assertNotIn(name, pins, f'duplicate requirement: {name}')
            pins[name] = version
        self.assertEqual({name: pins[name] for name in (
            'numpy', 'huggingface-hub', 'pillow', 'safetensors',
            'scipy', 'matplotlib', 'open3d')}, {
                'numpy': '2.4.4', 'huggingface-hub': '1.16.4', 'pillow': '11.2.1',
                'safetensors': '0.5.3', 'scipy': '1.16.0',
                'matplotlib': '3.10.3', 'open3d': '0.19.0',
            })
        self.assertNotIn('numpy==1.26.4', requirements)

    def test_failed_install_records_failure_and_does_not_claim_success(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            (root / 'requirements-base.txt').write_text((ROOT / 'requirements-base.txt').read_text())
            stack.enter_context(mock.patch.object(installer, 'ROOT', root))
            stack.enter_context(mock.patch.object(installer.platform, 'system', return_value='Linux'))
            stack.enter_context(mock.patch.object(installer.platform, 'machine', return_value='x86_64'))
            stack.enter_context(mock.patch.object(installer.platform, 'libc_ver', return_value=('glibc', '2.31')))
            stack.enter_context(mock.patch.object(installer.shutil, 'which', side_effect=lambda name: '/bin/' + name))
            stack.enter_context(mock.patch.object(installer, 'ensure_source', return_value=root / 'source'))
            stack.enter_context(mock.patch.object(installer, 'prepare_environment', return_value=(root / 'bin/python', ['dvlt'])))
            def fail_package_install(command, **kwargs):
                if 'pip' in command:
                    raise subprocess.CalledProcessError(1, command)
            process = stack.enter_context(mock.patch.object(installer.subprocess, 'run', side_effect=fail_package_install))
            freeze = stack.enter_context(mock.patch.object(installer.subprocess, 'check_output'))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(subprocess.CalledProcessError):
                    installer.install('dvlt', self.profiles, arguments())
            state = json.loads((root / 'reproducibility/environments/unimetric-dvlt/install-state.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertNotIn('Installed dvlt', output.getvalue())
            self.assertTrue(process.called)
            freeze.assert_not_called()

    def test_old_glibc_is_rejected_before_changes_or_processes(self):
        with mock.patch.object(installer.platform, 'system', return_value='Linux'), mock.patch.object(
            installer.platform, 'machine', return_value='x86_64'
        ), mock.patch.object(installer.platform, 'libc_ver', return_value=('glibc', '2.28')), mock.patch.object(
            installer.subprocess, 'run'
        ) as process, mock.patch.object(installer, 'ensure_source') as source, mock.patch.object(
            installer, 'prepare_environment'
        ) as environment:
            with self.assertRaisesRegex(RuntimeError, 'glibc >= 2.31'):
                installer.install('dvlt', self.profiles, arguments())
            process.assert_not_called()
            source.assert_not_called()
            environment.assert_not_called()

    def test_existing_mismatched_or_dirty_source_is_never_checked_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = dict(self.profiles['models']['dvlt']['source'])
            target = root / source['directory']
            (target / '.git').mkdir(parents=True)
            for revision, status in [('0' * 40, ''), (source['revision'], ' M user.py')]:
                with self.subTest(revision=revision, status=status):
                    with mock.patch.object(installer, 'ROOT', root), mock.patch.object(
                        installer, 'git_output', side_effect=[str(target), revision, status]
                    ), mock.patch.object(installer.subprocess, 'run') as process:
                        with self.assertRaises(RuntimeError):
                            installer.ensure_source(source)
                        process.assert_not_called()

    def test_existing_unowned_environment_requires_explicit_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            with mock.patch.object(installer, 'environment_prefix', return_value=prefix), mock.patch.object(
                installer.subprocess, 'check_output', return_value='3.11\n'
            ) as check, mock.patch.object(installer.subprocess, 'run') as process:
                with self.assertRaisesRegex(RuntimeError, '--reuse-env'):
                    installer.prepare_environment('conda', 'existing', ['dvlt'], self.profiles, False)
                check.assert_not_called()
                self.assertFalse((prefix / '.unimetric-environment.json').exists())
                python, members = installer.prepare_environment('conda', 'existing', ['dvlt'], self.profiles, True)
                self.assertEqual(python, prefix / 'bin/python')
                self.assertEqual(members, ['dvlt'])
                self.assertEqual(json.loads((prefix / '.unimetric-environment.json').read_text())['models'], ['dvlt'])
                process.assert_not_called()

    def test_common_and_all_expand_once_into_two_environment_groups(self):
        common = installer.select_models(['common'], self.profiles)
        self.assertEqual(set(common), set(self.profiles['models']) - {'dvlt'})
        self.assertEqual(len(common), 8)
        self.assertEqual(installer.select_models(['pi3', 'common', 'pi3_long'], self.profiles),
                         list(dict.fromkeys(['pi3', *common])))
        selected = installer.select_models(['common', 'all', 'dvlt_long'], self.profiles)
        self.assertEqual(len(selected), 9)
        with mock.patch.object(installer, 'install_group') as install_group:
            args = arguments()
            installer.install(selected, self.profiles, args)
        self.assertEqual(install_group.call_args_list, [
            mock.call(common, self.profiles, args),
            mock.call(['dvlt'], self.profiles, args),
        ])
        for model in common:
            profile = self.profiles['models'][model]
            self.assertEqual((profile['group'], profile['environment'], profile['python'],
                              profile['torch'], profile['torchvision'], profile['cuda']),
                             ('common', 'unimetric', '3.11', '2.6.0', '0.21.0', 'cu124'))
            self.assertIn('numpy==1.26.4', installer.merged_requirements(profile))

    def test_custom_environment_name_rejects_mixed_groups_before_install(self):
        with mock.patch.object(installer, 'install_group') as install_group:
            with self.assertRaisesRegex(ValueError, 'one environment group'):
                installer.install(['pi3', 'dvlt'], self.profiles, arguments(env_name='custom'))
            install_group.assert_not_called()

    def test_append_reuses_owned_and_legacy_markers_without_explicit_reuse(self):
        markers = [
            {'group': 'common', 'models': ['pi3', 'da3']},
            {'model': 'pi3'},
        ]
        for previous in markers:
            with self.subTest(previous=previous), tempfile.TemporaryDirectory() as directory:
                prefix = Path(directory)
                marker = prefix / '.unimetric-environment.json'
                marker.write_text(json.dumps({'repository': str(installer.ROOT), **previous}))
                expected = [*previous.get('models', ['pi3']), 'vggt']
                with mock.patch.object(installer, 'environment_prefix', return_value=prefix), mock.patch.object(
                    installer.subprocess, 'check_output', return_value='3.11\n'
                ), mock.patch.object(installer.subprocess, 'run') as process:
                    python, members = installer.prepare_environment(
                        'conda', 'unimetric', ['vggt'], self.profiles, False)
                self.assertEqual(python, prefix / 'bin/python')
                self.assertEqual(members, expected)
                self.assertEqual(json.loads(marker.read_text()), {
                    'repository': str(installer.ROOT), 'group': 'common', 'models': expected})
                process.assert_not_called()

    def test_explicit_reuse_never_crosses_recorded_environment_group(self):
        cases = [
            ({'group': 'common', 'models': ['pi3']}, ['dvlt']),
            ({'group': 'dvlt', 'models': ['dvlt']}, ['pi3']),
            ({'model': 'pi3'}, ['dvlt']),
            ({'model': 'dvlt'}, ['pi3']),
        ]
        for previous, selected in cases:
            with self.subTest(previous=previous), tempfile.TemporaryDirectory() as directory:
                prefix = Path(directory)
                marker = prefix / '.unimetric-environment.json'
                original = json.dumps({'repository': str(installer.ROOT), **previous})
                marker.write_text(original)
                with mock.patch.object(installer, 'environment_prefix', return_value=prefix), mock.patch.object(
                    installer.subprocess, 'check_output'
                ) as check, mock.patch.object(installer.subprocess, 'run') as process:
                    with self.assertRaisesRegex(RuntimeError, 'separate environments'):
                        installer.prepare_environment('conda', 'existing', selected, self.profiles, True)
                check.assert_not_called()
                process.assert_not_called()
                self.assertEqual(marker.read_text(), original)

    def test_append_jointly_installs_and_checks_all_but_downloads_only_new_model(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            (root / 'requirements-base.txt').write_text((ROOT / 'requirements-base.txt').read_text())
            prefix = root / 'envs/unimetric'
            prefix.mkdir(parents=True)
            (prefix / '.unimetric-environment.json').write_text(json.dumps({
                'repository': str(root), 'group': 'common', 'models': ['da3']}))
            stack.enter_context(mock.patch.object(installer, 'ROOT', root))
            stack.enter_context(mock.patch.object(installer.platform, 'system', return_value='Linux'))
            stack.enter_context(mock.patch.object(installer.platform, 'machine', return_value='x86_64'))
            stack.enter_context(mock.patch.object(installer.platform, 'libc_ver', return_value=('glibc', '2.31')))
            stack.enter_context(mock.patch.object(installer.shutil, 'which', side_effect=lambda name: '/bin/' + name))
            stack.enter_context(mock.patch.object(installer, 'environment_prefix', return_value=prefix))
            stack.enter_context(mock.patch.object(installer, 'ensure_source', side_effect=lambda source: root / source['directory']))
            stack.enter_context(mock.patch.object(installer.subprocess, 'check_output', side_effect=['3.11\n', 'mock-freeze\n']))
            process = stack.enter_context(mock.patch.object(installer.subprocess, 'run'))
            with contextlib.redirect_stdout(io.StringIO()):
                installer.install(['pi3'], self.profiles, arguments())
            commands = [call.args[0] for call in process.call_args_list]
            joint = [command for command in commands if '-e' in command]
            self.assertEqual(len(joint), 1)
            for model in ('da3', 'pi3'):
                source = str(root / self.profiles['models'][model]['source']['directory'])
                self.assertIn(source, joint[0])
            self.assertNotIn(str(root / self.profiles['models']['pi3']['source']['directory'] / 'requirements.txt'), joint[0])
            downloads = [command for command in commands if str(root / 'scripts/download_gfm_weights.py') in command]
            self.assertEqual([command[2] for command in downloads], ['pi3'])
            checks = [command for command in commands if str(root / 'scripts/check_gfm.py') in command]
            self.assertEqual([command[2] for command in checks], ['da3', 'pi3'])
            self.assertNotIn('--weights', checks[0])
            self.assertIn('--weights', checks[1])
            state_dir = root / 'reproducibility/environments/unimetric'
            requirements = (state_dir / 'requirements.txt').read_text().splitlines()
            self.assertIn('xformers==0.0.29.post3', requirements)
            cuda_installs = [command for command in commands if '--index-url' in command]
            self.assertEqual(len(cuda_installs), 1)
            self.assertIn('xformers==0.0.29.post3', cuda_installs[0])
            self.assertIn('https://download.pytorch.org/whl/cu124', cuda_installs[0])
            state = json.loads((state_dir / 'install-state.json').read_text())
            self.assertEqual(state['status'], 'installed')
            self.assertEqual(state['models'], ['da3', 'pi3'])
            self.assertEqual(state['selected_models'], ['pi3'])
            self.assertEqual((state_dir / 'pip-freeze.txt').read_text(), 'mock-freeze\n')

    def test_downloads_only_selected_files_and_preserves_existing_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = self.profiles['models']['dvlt']['weights']
            existing = root / weights['directory'] / 'model.safetensors'
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b'user-owned-checkpoint')
            download = mock.Mock()
            with mock.patch.object(downloader, 'ROOT', root), mock.patch.dict(
                sys.modules, {'huggingface_hub': SimpleNamespace(hf_hub_download=download)}
            ), mock.patch.object(sys, 'argv', ['download_gfm_weights.py', 'dvlt_long']), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(downloader.main(), 0)
            sam = self.profiles['sam2_weights']
            self.assertEqual(download.call_args_list, [
                mock.call(repo_id=sam['repo_id'], filename='sam2.1_hiera_large.pt',
                          revision=sam['revision'], local_dir=root / sam['directory']),
                mock.call(repo_id=weights['repo_id'], filename='config.json',
                          revision=weights['revision'], local_dir=root / weights['directory']),
            ])
            self.assertEqual(existing.read_bytes(), b'user-owned-checkpoint')


if __name__ == '__main__':
    unittest.main()

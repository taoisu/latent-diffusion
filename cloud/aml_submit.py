import fire

from azureml.core import (
    Environment,
    ScriptRunConfig,
    Experiment,
    Workspace,
    Datastore,
    Run,
)
from azureml.data.data_reference import DataReference
from azureml.core.runconfig import DockerConfiguration


def main(
    cluster:str='DS-C8-M28',
    task:str='prep_laion2b_get'
):
    docker_config = DockerConfiguration(
        use_docker=True,
        shm_size='256g',
    )
    ws = Workspace.from_config()
    exp = Experiment(ws, 'vdipt')
    cluster = ws.compute_targets[cluster]
    env = Environment.get(ws, name='vdipainter')
    dsf_avid = DataReference(
        Datastore(ws, 'airocrhbi_avidxchange_data'),
        data_reference_name='avid',
        path_on_compute='/mount/avid')
    dsf_ocrd = DataReference(
        Datastore(ws, 'airocrhbi_ocr_data'),
        data_reference_name='ocrd',
        path_on_compute='/mount/ocrd')
    scripts = {
        'prep_laion2b_get': 'cloud/entry_laion2b_get.sh',
        'prep_laion2b_map': 'cloud/entry_laion2b_map.sh',
        'prep_avid_gen': 'cloud/entry_avid_gen.sh',
        'prep_avid_flt': 'cloud/entry_avid_filter.sh',
        'prep_avid_gen_ocr': 'cloud/entry_avid_gen_ocr.sh',
        'train_avid_sr_128x512': 'cloud/entry_avid_train_sr_128x512.sh',
        'train_avid_inpaint_128': 'cloud/entry_avid_train_inpaint_128.sh'
    }
    script = scripts[task]
    config = ScriptRunConfig(
        source_directory='..',
        command=['bash', script],
        compute_target=cluster,
        environment=env,
        docker_runtime_config=docker_config)
    config.run_config.data_references = {
        dsf_avid.data_reference_name: dsf_avid.to_config(),
        dsf_ocrd.data_reference_name: dsf_ocrd.to_config(),
    }
    script_run:Run = exp.submit(config)
    print(script_run._run_details_url)


if __name__ == '__main__':
    fire.Fire(main)
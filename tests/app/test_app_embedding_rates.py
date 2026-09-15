'''The actual Streamlit settings controls persist provider-specific quotas.'''

from osintgpt import Project
from osintgpt.app.session import VIEW
from tests.app.test_app_renders import home, project, run


def test_embedding_quota_controls_save_and_remain_separate(project):
    app = run(project, **{VIEW: 'Settings'})
    assert not app.exception
    app.number_input(key='openai_embedding_tpm').set_value(2_000_000)
    app.number_input(key='openai_embedding_rpm').set_value(5000)
    app.number_input(key='openai_embedding_batch_tokens').set_value(50000)
    app.text_input(key='openai_embedding_quota_scope').set_value('shared-openai')
    next(button for button in app.button if button.label == 'Save').click().run()
    assert not app.exception
    saved = Project.load(project.paths.root)
    assert saved.settings.openai_embedding_tpm == 2_000_000
    assert saved.settings.openai_embedding_rpm == 5000
    assert saved.settings.openai_embedding_batch_tokens == 50000
    assert saved.settings.openai_embedding_quota_scope == 'shared-openai'
    next(select for select in app.selectbox if select.label == 'Embedding provider').select('gemini').run()
    assert app.number_input(key='gemini_embedding_tpm').value is None
    assert not app.exception

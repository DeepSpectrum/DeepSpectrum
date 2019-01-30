import tensorflow as tf
import tfplot
from ...tools.performance_stats import plot_confusion_matrix

class EvalConfusionMatrixHook(tf.train.SessionRunHook):
    

    def __init__(self,
                 output_dir=None,
                 summary_writer=None,
                 vocabulary=None):

        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._summary = None
        self._vocabulary = vocabulary


    def begin(self):
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = tf.summary.FileWriterCache.get(self._output_dir)
        self._next_step = None
        self._global_step_tensor = tf.train.get_or_create_global_step(
        )  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use SummarySaverHook.")
        op_name = [
            n.name for n in tf.get_default_graph().as_graph_def().node
            if n.name.endswith('confusion_matrix/sum')
        ][0]
        con_matrix_tensor = tf.get_default_graph().get_tensor_by_name(
            op_name + ':0')
        self._summary_op = tfplot.summary.plot(
            'cm_plot',
            plot_confusion_matrix, [con_matrix_tensor],
            normalize=True,
            classes=self._vocabulary)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        requests = {"global_step": self._global_step_tensor}
        if self._get_summary_op() is not None:
            requests["summary"] = self._get_summary_op()

        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        _ = run_context
        if not self._summary_writer:
            return

        global_step = run_values.results["global_step"]
        if self._next_step is None:
            global_step = run_context.session.run(self._global_step_tensor)

        if self._next_step is None:
            self._summary_writer.add_session_log(
                tf.SessionLog(status=tf.SessionLog.START), global_step)

        if "summary" in run_values.results:
            self._summary = run_values.results["summary"]

        self._next_step = global_step + 1

    def end(self, session=None):
        for summary in self._summary:
            self._summary_writer.add_summary(summary, self._next_step - 1)
        if self._summary_writer:
            self._summary_writer.flush()

    def _get_summary_op(self):
        """Fetches the summary op either from self._summary_op or self._scaffold.
        Returns:
        Returns a list of summary `Tensor`.
        """
        summary_op = None
        if self._summary_op is not None:
            summary_op = self._summary_op
        elif self._scaffold.summary_op is not None:
            summary_op = self._scaffold.summary_op

        if summary_op is None:
            return None

        if not isinstance(summary_op, list):
            return [summary_op]
        return summary_op



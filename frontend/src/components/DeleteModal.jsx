export default function DeleteModal({ doc, onArchive, onPermanent, onCancel }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
      onClick={onCancel}
    >
      <div
        className="
          bg-white dark:bg-slate-800
          rounded-2xl shadow-2xl
          max-w-sm w-full mx-4 p-6
          flex flex-col gap-4
        "
        onClick={e => e.stopPropagation()}
      >
        <div>
          <h2 className="text-base font-semibold text-slate-800 dark:text-slate-100">
            Remove document?
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 truncate">
            {doc}
          </p>
        </div>

        <div className="flex flex-col gap-2">
          <button
            onClick={onArchive}
            className="
              w-full text-left rounded-xl px-4 py-3
              bg-slate-50 dark:bg-slate-700
              hover:bg-slate-100 dark:hover:bg-slate-600
              transition-colors
            "
          >
            <p className="text-sm font-medium text-slate-800 dark:text-slate-100">
              Archive
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Hidden from search — restore later for free
            </p>
          </button>

          <button
            onClick={onPermanent}
            className="
              w-full text-left rounded-xl px-4 py-3
              bg-red-50 dark:bg-red-900/20
              hover:bg-red-100 dark:hover:bg-red-900/40
              transition-colors
            "
          >
            <p className="text-sm font-medium text-red-600 dark:text-red-400">
              Delete permanently
            </p>
            <p className="text-xs text-red-400 dark:text-red-500 mt-0.5">
              Removes all data — re-upload will reprocess
            </p>
          </button>
        </div>

        <button
          onClick={onCancel}
          className="
            text-xs text-slate-400 hover:text-slate-600
            dark:hover:text-slate-200
            transition-colors self-center
          "
        >
          Cancel
        </button>
      </div>
    </div>
  )
}
